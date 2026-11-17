"""Pretraining loop (single GPU or multi-GPU DistributedDataParallel).

AdamW (betas 0.9/0.95, weight decay 0.1 except biases, norms and latent tokens),
linear warm-up over the first 2% of optimizer steps then cosine decay to 1e-5,
bfloat16 autocast, gradient clipping at 1.0, two micro-batches per step.
An epoch draws as many cells as the training set (with replacement,
species-balanced); the schedule spans ``max_epochs`` epochs.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import DataConfig, LossConfig, ModelConfig, TrainConfig
from .corpus import BucketedWeightedBatchSampler, Corpus, PackedCells
from .data import PointCollator
from .families import FamilySpace, GenomeIndex
from .genes import GeneTable
from .loss import LossModule, masked_nll
from .model import ScFLORA


def lr_at(step: int, total: int, t: TrainConfig) -> float:
    w = max(1, int(total * t.warmup_frac))
    if step < w:
        return t.lr * (step + 1) / w
    p = min(1.0, (step - w) / max(1, total - w))
    return t.min_lr + 0.5 * (t.lr - t.min_lr) * (1 + math.cos(math.pi * p))


def save_checkpoint(path: str, model: ScFLORA, cfgs: Dict, step: int, extra: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfgs, "step": int(step),
                "format": "scflora-v1", **(extra or {})}, path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def evaluate(model, loader, sampler, lcfg, dcfg, dev, max_batches: int, amp_dtype) -> Dict[str, float]:
    """Validation NLL with fixed masking seeds (comparable across steps)."""
    model.eval()
    acc, n = {}, 0
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        b = b.to(dev, non_blocking=True)
        g = torch.Generator(device=dev).manual_seed(bi)
        ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else torch.autocast("cuda", enabled=False)
        with ctx:
            _, st = masked_nll(model, b, sampler, lcfg, dcfg, generator=g)
        for k, v in st.items():
            acc[k] = acc.get(k, 0.0) + v
        n += 1
    model.train()
    return {f"val/{k}": v / max(n, 1) for k, v in acc.items()}


def build_index(gene_dir: str, species: List[str], family_space: str) -> GenomeIndex:
    tables = [GeneTable.load(os.path.join(gene_dir, f"{s}.npz")) for s in species]
    return GenomeIndex(tables, FamilySpace.load(family_space))


def train(args, mcfg: ModelConfig, dcfg: DataConfig, lcfg: LossConfig, tcfg: TrainConfig) -> None:
    devs = [int(x) for x in str(args.devices).split(",") if x != ""]
    if len(devs) <= 1:
        _worker(0, 1, devs, args, mcfg, dcfg, lcfg, tcfg)
        return
    import torch.multiprocessing as mp
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 2000))
    mp.spawn(_worker, args=(len(devs), devs, args, mcfg, dcfg, lcfg, tcfg), nprocs=len(devs), join=True)


def _worker(rank, world, devs, args, mcfg, dcfg, lcfg, tcfg) -> None:
    import torch.distributed as dist
    main = rank == 0
    dev = torch.device(f"cuda:{devs[rank]}" if devs else "cuda:0")
    torch.cuda.set_device(dev)
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.set_float32_matmul_precision("high")          # TF32 matmuls, as in pretraining
    torch.manual_seed(tcfg.seed + rank); np.random.seed(tcfg.seed + rank)
    cfgs = {"model": asdict(mcfg), "data": asdict(dcfg), "loss": asdict(lcfg), "train": asdict(tcfg)}
    os.makedirs(args.out_dir, exist_ok=True)
    if main:
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(cfgs, f, indent=2)

    species = [l.strip() for l in open(args.species) if l.strip()]
    index = build_index(args.gene_dir, species, args.family_space)
    corpus = Corpus(args.manifest, index, os.path.join(args.out_dir, "cache"), dcfg, seed=tcfg.seed,
                    col2tok_dir=args.col2tok_dir, verbose=main)
    if args.exclude_mask:
        corpus.exclude(args.exclude_mask)
    tr_idx, va_idx = corpus.train_idx, corpus.val_idx
    coll = PointCollator()
    bs = BucketedWeightedBatchSampler(corpus.sample_weights(tr_idx, tcfg.species_alpha), corpus.nnz[tr_idx],
                                      tcfg.batch_size, len(tr_idx), dcfg.bucket_mult, tcfg.seed,
                                      dcfg.max_tokens, dcfg.max_batch, rank, world)
    kw = dict(num_workers=dcfg.n_workers, pin_memory=dcfg.pin_memory, collate_fn=coll)
    if dcfg.n_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=dcfg.prefetch_factor)
    tr = torch.utils.data.DataLoader(PackedCells(corpus, args.pack_dir, tr_idx), batch_sampler=bs, **kw)
    va = torch.utils.data.DataLoader(PackedCells(corpus, args.pack_dir, va_idx), batch_size=tcfg.batch_size,
                                     shuffle=False, num_workers=(max(1, dcfg.n_workers // 2) if dcfg.n_workers else 0),
                                     collate_fn=coll)

    model = ScFLORA(mcfg, index.space).to(dev)
    model.set_genome(index, dev)
    sampler = index.family_sampler(lcfg.family_samples).to(dev)
    wrapped = LossModule(model, sampler, lcfg, dcfg)
    net = (torch.nn.parallel.DistributedDataParallel(wrapped, device_ids=[dev.index], broadcast_buffers=False)
           if world > 1 else wrapped)
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or n_ == "latents" else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tcfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tcfg.lr, betas=tuple(tcfg.betas))
    accum = max(1, tcfg.accumulate)
    nb = torch.tensor([len(bs) // accum], device=dev)
    if world > 1:
        dist.all_reduce(nb, op=dist.ReduceOp.MIN)
    steps_per_epoch = int(nb.item())
    total = steps_per_epoch * tcfg.max_epochs
    total_run = min(total, args.max_steps) if args.max_steps else total
    amp = torch.bfloat16 if tcfg.precision == "bf16" else None
    gen = torch.Generator(device=dev).manual_seed(tcfg.seed + 7919 * rank)

    step, start_ep, start_bi = 0, 0, 0
    resume = os.path.join(args.out_dir, "resume.pt")
    if os.path.exists(resume):
        st = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["state_dict"]); opt.load_state_dict(st["opt"])
        step, start_ep, start_bi = st["step"], st["epoch"], st["next_bi"]
        if main:
            print(f"[resume] step {step:,}, epoch {start_ep}", flush=True)
    if main:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"[train] {n_par / 1e6:.2f} M trainable parameters; {steps_per_epoch:,} steps/epoch x "
              f"{tcfg.max_epochs} epochs = {total:,} steps", flush=True)

    net.train()
    t0 = time.time()
    for ep in range(start_ep, tcfg.max_epochs):
        bs.set_epoch(ep)
        bi0 = start_bi if ep == start_ep else 0
        bs.skip = bi0
        for bi, batch in enumerate(tr, start=bi0):
            if bi >= steps_per_epoch * accum or step >= total_run:
                break
            batch = batch.to(dev, non_blocking=True)
            last = bi % accum == accum - 1
            if last:
                for g_ in opt.param_groups:
                    g_["lr"] = lr_at(step, total, tcfg)
            ctx = torch.autocast("cuda", dtype=amp) if amp else torch.autocast("cuda", enabled=False)
            sync = net.no_sync() if (world > 1 and not last) else contextlib.nullcontext()
            with sync:
                with ctx:
                    loss, st = net(batch, generator=gen)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}: {st}")
                (loss / accum).backward()
            if not last:
                continue
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if main and step % tcfg.log_every == 0:
                print(f"[{step:,}/{total:,}] nll={st['nll']:.4f} |g|={float(gn):.2f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s", flush=True)
            if tcfg.save_every and step % tcfg.save_every == 0:
                if main:
                    vm = evaluate(model, va, sampler, lcfg, dcfg, dev, tcfg.limit_val_batches, amp)
                    print("  [val] " + " ".join(f"{k}={v:.4f}" for k, v in vm.items()), flush=True)
                    save_checkpoint(os.path.join(args.out_dir, f"step{step}.pt"), model, cfgs, step, {"val": vm})
                    torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(), "step": step,
                                "epoch": ep, "next_bi": bi + 1}, resume + ".tmp")
                    os.replace(resume + ".tmp", resume)
                if world > 1:
                    dist.barrier()
        if step >= total_run:
            break
    if main:
        vm = evaluate(model, va, sampler, lcfg, dcfg, dev, tcfg.limit_val_batches, amp)
        save_checkpoint(os.path.join(args.out_dir, "last.pt"), model, cfgs, step, {"val": vm})
        print(f"[train] done: {step:,} steps in {(time.time() - t0) / 3600:.2f} h", flush=True)
    if world > 1:
        dist.barrier(); dist.destroy_process_group()
