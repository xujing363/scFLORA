"""ESM-2 (3B) protein embeddings for a proteome, one vector per gene.

Recipe used for every species in this work:
  * longest protein isoform per gene (FASTA header -> gene id by a user function);
  * HuggingFace ``facebook/esm2_t36_3B_UR50D`` in float16, sequences truncated to
    1,022 residues;
  * last hidden layer mean-pooled over all tokens covered by the attention mask
    (including the BOS/EOS tokens), stored as float16.
"""
from __future__ import annotations

import gzip
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

MODEL_NAME = "facebook/esm2_t36_3B_UR50D"


def longest_isoforms(fasta: str, gene_of_header: Optional[Callable[[str], str]] = None
                     ) -> List[Tuple[str, str]]:
    """[(gene_id, sequence)] keeping the longest protein per gene, sorted by gene id.
    By default the gene id is the first word of the header."""
    fn = gene_of_header or (lambda h: h.split()[0])
    best: Dict[str, str] = {}
    gid, chunks = None, []
    op = gzip.open if fasta.endswith(".gz") else open

    def flush():
        if gid is None:
            return
        s = "".join(chunks).rstrip("*")
        if s and (gid not in best or len(s) > len(best[gid])):
            best[gid] = s

    with op(fasta, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                gid, chunks = fn(line[1:]), []
            else:
                chunks.append(line.strip())
    flush()
    return sorted(best.items())


@torch.no_grad()
def embed_sequences(seqs: List[str], device: str = "cuda:0", token_budget: int = 16000,
                    max_len: int = 1022, model_name: str = MODEL_NAME, verbose: bool = True) -> np.ndarray:
    """[N, 2560] float16 mean-pooled embeddings, in input order (length-sorted batches)."""
    from transformers import AutoTokenizer, EsmModel
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = EsmModel.from_pretrained(model_name, torch_dtype=torch.float16, add_pooling_layer=False).eval().to(device)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = np.empty((len(seqs), mod.config.hidden_size), dtype=np.float16)
    i, t0 = 0, time.time()
    while i < len(order):
        j, width = i, 0
        while j < len(order):
            w = max(width, min(len(seqs[order[j]]), max_len) + 2)
            if j > i and w * (j - i + 1) > token_budget:
                break
            width = w; j += 1
        idx = order[i:j]
        enc = tok([seqs[k][:max_len] for k in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len + 2)
        enc = {k: v.to(device) for k, v in enc.items()}
        h = mod(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        out[idx] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy().astype(np.float16)
        i = j
        if verbose and (i == len(order) or i % 5000 < len(idx)):
            print(f"  {i:,}/{len(seqs):,} proteins, {i / max(time.time() - t0, 1e-9):.1f}/s", flush=True)
    return out


def embed_proteome(fasta: str, out_npz: str, device: str = "cuda:0",
                   gene_of_header: Optional[Callable[[str], str]] = None,
                   model_name: str = MODEL_NAME) -> None:
    """Write ``out_npz`` with ``gene_ids`` and ``emb`` ([N, 2560] float16)."""
    items = longest_isoforms(fasta, gene_of_header)
    emb = embed_sequences([s for _, s in items], device=device, model_name=model_name)
    np.savez(out_npz, gene_ids=np.array([g for g, _ in items]), emb=emb)
