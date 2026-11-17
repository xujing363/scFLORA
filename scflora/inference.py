"""Inference with a pretrained scFLORA checkpoint: cell embeddings, genome-wide
predicted expression and virtual gene knockouts.

All functions run the frozen model in evaluation mode; nothing is fine-tuned.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, from_dict
from .data import MatrixPoints, PairedKnockoutPoints, PointCollator, paired_collate
from .families import FamilySpace, GenomeIndex
from .genes import GeneTable
from .model import ScFLORA


def load_pretrained(checkpoint: str, family_space: str, device: str = "cuda:0") -> ScFLORA:
    """Load released weights. ``family_space`` is the ``family_space.npz`` asset.

    Sets ``torch.set_float32_matmul_precision("high")`` (TF32 matmuls on GPU), the
    numerics used for pretraining and for all analyses in the paper.
    """
    torch.set_float32_matmul_precision("high")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = from_dict(ModelConfig, ck["config"]["model"])
    model = ScFLORA(cfg, FamilySpace.load(family_space))
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing {missing}, unexpected {unexpected}")
    return model.eval().to(device)


def prepare_species(model: ScFLORA, table: Union[GeneTable, str], species: Optional[str] = None
                    ) -> GeneTable:
    """Attach one species to the model.

    ``table`` is a released gene table (``genes/<species>.npz``; families as in
    pretraining) or an ESM-2 embedding file of a new species, whose genes are
    assigned to the nearest family centroid.
    """
    if isinstance(table, str):
        z = np.load(table, allow_pickle=True) if table.endswith(".npz") else None
        if z is not None and "family" in z.files:
            table = GeneTable.load(table)
        else:
            if species is None:
                raise ValueError("species name required for a new ESM-2 embedding file")
            table = GeneTable.from_esm_file(table, species)
    if table.family is None:
        table.family = _space_of(model).assign(table.esm, device=str(model.log_beta.device))
    index = GenomeIndex([table], _space_of(model))
    model.set_genome(index)
    return table


def _space_of(model: ScFLORA) -> FamilySpace:
    sp = getattr(model, "_family_space", None)
    if sp is None:
        raise RuntimeError("model was not loaded with load_pretrained()")
    return sp


def _autocast(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


@torch.no_grad()
def embed(model: ScFLORA, X, var_names: Sequence[str], batch_size: int = 64,
          num_workers: int = 4, amp: bool = True, genes: Optional[Sequence[str]] = None,
          support: str = "genome") -> np.ndarray:
    """512-d embeddings of the cells (rows) of raw-count matrix ``X``.

    ``genes`` optionally restricts the input to a gene panel (e.g. highly
    variable genes); counts of other genes are ignored.
    ``support``: support of the encoder's predicted family measure, ``"genome"``
    (all families of the species, as in pretraining; default) or ``"input"``
    (families of the input genes, used for the cross-species integration
    benchmarks with species-wise HVG panels).
    """
    if model.genome is None:
        raise RuntimeError("call prepare_species() first")
    col2tok = model.genome.tokens_of(model.genome.species[0], var_names)
    if genes is not None:
        keep = np.isin(np.asarray(var_names).astype(str), np.asarray(genes).astype(str))
        col2tok = np.where(keep, col2tok, -1)
    if (col2tok >= 0).sum() == 0:
        raise ValueError("no gene of the matrix is in the species gene table; check gene ids/species")
    if support == "input":
        model.set_measure_support(model.genome.fine_of_tok[np.unique(col2tok[col2tok >= 0])])
    elif support == "genome":
        model.set_measure_support(None)
    else:
        raise ValueError("support must be 'genome' or 'input'")
    ds = MatrixPoints(X, col2tok)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    collate_fn=PointCollator())
    dev = model.log_beta.device
    out = []
    for b in dl:
        b = b.to(dev, non_blocking=True)
        with _autocast(dev, amp):
            out.append(model(b)["cell_emb"].float().cpu())
    E = torch.cat(out).numpy().astype(np.float32)
    if not np.isfinite(E).all():
        raise RuntimeError("non-finite embeddings")
    return E


def embed_adata(model: ScFLORA, adata, layer: Optional[str] = None, key_added: str = "X_scflora",
                **kw) -> np.ndarray:
    """Embed an AnnData with raw counts in ``adata.X`` (or ``adata.layers[layer]``)."""
    X = adata.X if layer is None else adata.layers[layer]
    E = embed(model, X, adata.var_names.astype(str), **kw)
    adata.obsm[key_added] = E
    return E


@torch.no_grad()
def predict_rates(model: ScFLORA, X, var_names: Sequence[str], batch_size: int = 16,
                  chunk: int = 1024) -> Dict[str, np.ndarray]:
    """Genome-wide predicted rates (expected counts at each cell's depth) for every
    protein-coding gene of the attached species. Returns {'genes', 'rates' [cells, genes]}."""
    col2tok = model.genome.tokens_of(model.genome.species[0], var_names)
    dl = DataLoader(MatrixPoints(X, col2tok), batch_size=batch_size, shuffle=False,
                    collate_fn=PointCollator())
    dev = model.log_beta.device
    rates, gtok = [], None
    with torch.autocast(dev.type, enabled=False):
        for b in dl:
            b = b.to(dev)
            o = model(b)
            ll, gtok = model.log_rates(o["z"].float(), o["depth_eff"].float(), chunk=chunk)
            rates.append(torch.exp(ll).cpu().numpy())
    names = model.genome.gene_ids[0][gtok.cpu().numpy()]
    return {"genes": names, "rates": np.concatenate(rates)}


@torch.no_grad()
def virtual_knockout(model: ScFLORA, X, var_names: Sequence[str], targets: Sequence[str],
                     candidates: Optional[Sequence[str]] = None, batch_size: int = 16,
                     chunk: int = 1024, num_workers: int = 4) -> Dict[str, np.ndarray]:
    """In-silico knockout on wild-type cells.

    For each cell, the target genes' counts are set to zero (removed from the
    point set) and the original and modified cells are encoded separately.
    Two readouts are returned:

    * ``score`` [n_candidates]: mean over cells of |s_KO(g) - s_WT(g)|, the change in
      the decoder's within-family allocation logit (used to rank responsive genes);
    * ``log2_response`` [n_genes]: log2 of sum_i lambda_ig(KO) / sum_i lambda_ig(WT), the
      signed genome-wide predicted response at the wild-type depth.

    Computed in float32.
    """
    gen = model.genome
    col2tok = gen.tokens_of(gen.species[0], var_names)
    tg = gen.tokens_of(gen.species[0], targets)
    if (tg < 0).any():
        missing = [t for t, k in zip(targets, tg) if k < 0]
        raise ValueError(f"targets not in the gene table: {missing}")
    if candidates is None:
        cand = np.unique(col2tok[col2tok >= 0])
    else:
        cand = gen.tokens_of(gen.species[0], candidates)
        cand = np.unique(cand[cand >= 0])
    cand = cand[~np.isin(cand, tg)]
    ds = PairedKnockoutPoints(X, col2tok, drop_tokens=tg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    collate_fn=paired_collate(PointCollator()))
    dev = model.log_beta.device
    score, n = np.zeros(len(cand)), 0
    s_wt = s_ko = None
    with torch.autocast(dev.type, enabled=False):
        for bw, bk in dl:
            bw, bk = bw.to(dev), bk.to(dev)
            ow, ok = model(bw), model(bk)
            zw, zk = ow["z"].float(), ok["z"].float()
            hw, hk = model.readout(zw).float(), model.readout(zk).float()
            sw, sk = model.decoder_slots(zw), model.decoder_slots(zk)
            parts = []
            for a in range(0, len(cand), chunk):
                t = torch.as_tensor(cand[a:a + chunk], device=dev)
                q = model.embed_tokens(t.unsqueeze(0).expand(zw.shape[0], -1)).float()
                parts.append(torch.abs(model.alloc_logit(model.deform(q, sk), hk)
                                       - model.alloc_logit(model.deform(q, sw), hw)).cpu().numpy())
            score += np.concatenate(parts, axis=1).sum(0)
            n += zw.shape[0]
            depth = ow["depth_eff"].float()          # wild-type depth for both arms
            lw, gtok = model.log_rates(zw, depth, chunk=chunk)
            lk, _ = model.log_rates(zk, depth, chunk=chunk)
            ew, ek = torch.exp(lw).double().sum(0).cpu().numpy(), torch.exp(lk).double().sum(0).cpu().numpy()
            s_wt = ew if s_wt is None else s_wt + ew
            s_ko = ek if s_ko is None else s_ko + ek
    ids = gen.gene_ids[0]
    return {"candidates": ids[cand], "score": score / max(n, 1),
            "genes": ids[gtok.cpu().numpy()], "log2_response": np.log2(s_ko / s_wt),
            "n_cells": n}
