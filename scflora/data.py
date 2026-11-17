"""Cells as variable-length point sets of (gene token, raw count).

No truncation, no padding token and no gene ordering: each cell contributes all
of its positive-count genes that are present in the species' gene table.
Counts must be raw (integer UMI) counts.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import scipy.sparse as sp
import torch

from .model import PointBatch


class PointCollator:
    """list of {tok, val, species} -> PointBatch (zero counts are dropped)."""

    def __call__(self, items: List[dict]) -> PointBatch:
        toks, vals = [], []
        for it in items:
            t = np.asarray(it["tok"], dtype=np.int64)
            v = np.asarray(it["val"], dtype=np.float32)
            keep = v > 0
            if keep.any():
                t, v = t[keep], v[keep]
            toks.append(t); vals.append(v)
        B, N = len(items), max(1, max(len(t) for t in toks))
        tok = np.zeros((B, N), dtype=np.int64)
        cnt = np.zeros((B, N), dtype=np.float32)
        valid = np.zeros((B, N), dtype=bool)
        for i, (t, v) in enumerate(zip(toks, vals)):
            tok[i, :len(t)] = t; cnt[i, :len(t)] = v; valid[i, :len(t)] = True
        return PointBatch(tok=torch.from_numpy(tok), cnt=torch.from_numpy(cnt),
                          valid=torch.from_numpy(valid),
                          species=torch.tensor([it["species"] for it in items], dtype=torch.long),
                          depth=torch.from_numpy(cnt.sum(1)))


def merge_duplicates(t: np.ndarray, v: np.ndarray):
    """Sum counts of columns that map to the same gene token."""
    if len(t) > 1:
        u, inv = np.unique(t, return_inverse=True)
        if len(u) != len(t):
            m = np.zeros(len(u), dtype=np.float32)
            np.add.at(m, inv, v)
            return u, m
    return t, v


class MatrixPoints(torch.utils.data.Dataset):
    """Rows of a cells x genes count matrix as point sets.

    col2tok maps each column to a token of the attached GenomeIndex (-1 = not in
    the gene table). ``drop_tokens`` removes genes from every cell (used for
    virtual knockouts).
    """

    def __init__(self, X, col2tok: np.ndarray, species: int = 0,
                 drop_tokens: Optional[np.ndarray] = None):
        self.X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        self.col2tok = np.asarray(col2tok, dtype=np.int64)
        if self.X.shape[1] != len(self.col2tok):
            raise ValueError("col2tok length must equal the number of columns")
        self.species = int(species)
        self.drop = None if drop_tokens is None else np.asarray(sorted(drop_tokens), dtype=np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def row(self, i: int, knockout: bool = False) -> dict:
        a, b = self.X.indptr[i], self.X.indptr[i + 1]
        t = self.col2tok[self.X.indices[a:b]]
        v = self.X.data[a:b].astype(np.float32, copy=False)
        keep = t >= 0
        t, v = t[keep], v[keep]
        if knockout and self.drop is not None and len(t):
            keep = ~np.isin(t, self.drop)
            t, v = t[keep], v[keep]
        t, v = merge_duplicates(t, v)
        return {"tok": t, "val": v, "species": self.species}

    def __getitem__(self, i: int) -> dict:
        return self.row(i)


class PairedKnockoutPoints(MatrixPoints):
    """(wild-type, knockout) pairs of the same cell."""

    def __getitem__(self, i: int):
        return self.row(i, False), self.row(i, True)


def paired_collate(collator: PointCollator):
    def f(items):
        return collator([x[0] for x in items]), collator([x[1] for x in items])
    return f
