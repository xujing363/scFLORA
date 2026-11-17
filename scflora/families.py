"""Protein-semantic families and genome indices.

Families are clusters of L2-normalized ESM-2 protein embeddings obtained by
spherical k-means over the pretraining vocabulary (30,000 fine families; 2,000
coarse strata used only for stratified sampling). A gene of any species is
assigned to the family with the nearest centroid, so the family space is
shared across species and no orthology mapping is used.

``GenomeIndex`` stacks the gene tables of one or more species into a single
token space and precomputes, per species, the families present in its genome
and the member lists of each (species, family) pair. These tables define the
family-level measure read by the encoder and the within-family allocation of
the decoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .genes import GeneTable


# ──────────────────────────────────────────────────────────────────────────────
# Family space
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FamilySpace:
    centers: np.ndarray          # [k_fine, 2560] float16, unit-norm family centroids
    coarse_of_fine: np.ndarray   # [k_fine] int64, coarse stratum of each family
    k_coarse: int

    @property
    def k_fine(self) -> int:
        return int(self.centers.shape[0])

    def save(self, path: str) -> None:
        np.savez(path, centers=self.centers.astype(np.float16),
                 coarse_of_fine=self.coarse_of_fine.astype(np.int64),
                 k_coarse=np.int64(self.k_coarse))

    @staticmethod
    def load(path: str) -> "FamilySpace":
        z = np.load(path)
        return FamilySpace(np.asarray(z["centers"]), np.asarray(z["coarse_of_fine"]).astype(np.int64),
                           int(z["k_coarse"]))

    @torch.no_grad()
    def assign(self, esm: np.ndarray, device: str = "cpu", chunk: int = 8192) -> np.ndarray:
        """Nearest-centroid (cosine) family of each row of ``esm`` (normalized float16,
        as stored in a GeneTable)."""
        dev = torch.device(device)
        C = F.normalize(torch.as_tensor(self.centers, dtype=torch.float32), dim=1)
        C = C.to(torch.float16).to(dev).float()
        out = np.empty(esm.shape[0], dtype=np.int64)
        for a in range(0, esm.shape[0], chunk):
            e = torch.as_tensor(np.asarray(esm[a:a + chunk], dtype=np.float16)).to(dev)
            x = F.normalize(e.float(), dim=1)
            out[a:a + chunk] = (x @ C.T).argmax(1).cpu().numpy()
        return out


def spherical_kmeans(X: Tensor, k: int, iters: int = 25, seed: int = 42,
                     chunk: int = 65536, verbose: bool = True) -> Tuple[Tensor, Tensor]:
    """Spherical k-means on unit-norm rows of ``X`` (random initialization).

    Returns (assignment [N], centers [k, D] unit-norm).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    N, D = X.shape
    C = X[torch.randperm(N, generator=g)[:k].to(X.device)].clone()
    while chunk * k * 4 > 6 * 1024 ** 3 and chunk > 1024:
        chunk //= 2
    assign = torch.zeros(N, dtype=torch.int64, device=X.device)
    for it in range(iters):
        moved = 0
        for a in range(0, N, chunk):
            b = min(a + chunk, N)
            new = (X[a:b] @ C.T).argmax(dim=1)
            moved += int((new != assign[a:b]).sum())
            assign[a:b] = new
        C_new = torch.zeros(k, D, dtype=torch.float32, device=X.device)
        C_new.index_add_(0, assign, X.float())
        cnt = torch.bincount(assign, minlength=k).float()
        empty = cnt == 0
        if empty.any():                                    # re-seed empty clusters
            idx = torch.randperm(N, generator=g)[:int(empty.sum())].to(X.device)
            C_new[empty] = X[idx].float()
            cnt[empty] = 1.0
        C = F.normalize(C_new / cnt.unsqueeze(1), dim=1).to(X.dtype)
        if verbose:
            print(f"  [kmeans k={k}] iter {it + 1}/{iters}: {moved:,} reassigned", flush=True)
        if moved == 0:
            break
    return assign, C


@torch.no_grad()
def build_family_space(tables: Sequence[GeneTable], k_coarse: int = 2000, k_fine: int = 30000,
                       iters: int = 25, seed: int = 42, device: str = "cuda:0"
                       ) -> Tuple[FamilySpace, List[GeneTable]]:
    """Cluster the ESM-2 embeddings of all genes of ``tables`` into coarse and fine
    families. Returns the family space and the tables with ``family`` filled in
    (the k-means assignment of the fine level)."""
    dev = torch.device(device)
    n = sum(len(t) for t in tables)
    X = torch.empty(n, tables[0].esm.shape[1], dtype=torch.float16, device=dev)
    o = 0
    for t in tables:                       # rows are already unit-norm float16
        X[o:o + len(t)] = torch.from_numpy(np.asarray(t.esm, dtype=np.float16)).to(dev)
        o += len(t)
    _, Cc = spherical_kmeans(X, k_coarse, iters, seed)
    af, Cf = spherical_kmeans(X, k_fine, iters, seed + 1)
    a = F.normalize(Cf.float(), dim=1)
    b = F.normalize(Cc.float(), dim=1)
    coarse_of_fine = (a @ b.T).argmax(1).cpu().numpy().astype(np.int64)
    af = af.cpu().numpy()
    out, o = [], 0
    for t in tables:
        out.append(GeneTable(t.species, t.gene_ids, t.esm, af[o:o + len(t)]))
        o += len(t)
    space = FamilySpace(Cf.cpu().numpy().astype(np.float16), coarse_of_fine, k_coarse)
    return space, out


# ──────────────────────────────────────────────────────────────────────────────
# Stratified importance sampler over a species' families
# ──────────────────────────────────────────────────────────────────────────────

class StratifiedSampler(nn.Module):
    """Unbiased stratified estimator of sums over a species' families.

    Families of species s are split into coarse strata J_j of size n_j; k_j
    families are drawn uniformly with replacement from J_j (k_j proportional to
    n_j, at least one, summing to K) and weighted by n_j / k_j. Then
    E[sum_j (n_j/k_j) sum_t x(f_jt)] = sum_f x(f), exactly.
    """

    def __init__(self, banks: Sequence[np.ndarray], stratum: np.ndarray, n_strata: int,
                 n_samples: int):
        super().__init__()
        self.K = int(n_samples)
        bank_all, bank_off, base_tab, size_tab, w_tab = [], [], [], [], []
        off = 0
        for s, bank in enumerate(banks):
            bank = np.asarray(bank, dtype=np.int64)
            cl = stratum[bank]
            order = np.argsort(cl, kind="stable")
            bank_sorted, cl_sorted = bank[order], cl[order]
            sizes = np.bincount(cl_sorted, minlength=n_strata).astype(np.int64)
            starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
            nz = np.flatnonzero(sizes > 0)
            if len(nz) > self.K:
                raise ValueError(f"genome {s}: {len(nz)} non-empty strata > {self.K} samples; "
                                 "the stratified estimator would be biased")
            sz = sizes[nz].astype(np.float64)
            alloc = np.maximum(1, np.floor(self.K * sz / sz.sum()).astype(np.int64))
            diff = int(self.K - alloc.sum())
            ordr, i = np.argsort(-sz), 0
            while diff != 0:
                j = ordr[i % len(ordr)]
                if diff > 0:
                    alloc[j] += 1; diff -= 1
                elif alloc[j] > 1:
                    alloc[j] -= 1; diff += 1
                i += 1
            base_tab.append(np.repeat(starts[nz], alloc))
            size_tab.append(np.repeat(sizes[nz], alloc))
            w_tab.append(np.repeat(sizes[nz] / alloc, alloc))
            bank_all.append(bank_sorted); bank_off.append(off)
            off += len(bank_sorted)
        self.register_buffer("bank_all", torch.from_numpy(np.concatenate(bank_all)).long())
        self.register_buffer("bank_off", torch.tensor(bank_off, dtype=torch.long))
        self.register_buffer("bank_len", torch.tensor([len(b) for b in bank_all], dtype=torch.long))
        self.register_buffer("slot_base", torch.from_numpy(np.stack(base_tab)).long())
        self.register_buffer("slot_size", torch.from_numpy(np.stack(size_tab)).long())
        self.register_buffer("slot_w", torch.from_numpy(np.stack(w_tab)).float())

    def sample(self, genome_id: Tensor, generator: Optional[torch.Generator] = None
               ) -> Tuple[Tensor, Tensor]:
        """Return (family ids [B, K], importance weights [B, K])."""
        B = genome_id.shape[0]
        base, size, w = self.slot_base[genome_id], self.slot_size[genome_id], self.slot_w[genome_id]
        u = torch.rand(B, self.K, device=self.bank_all.device, generator=generator)
        off = (u * size.to(u.dtype)).long().clamp_(max=size - 1)
        return self.bank_all[self.bank_off[genome_id].unsqueeze(1) + base + off], w


# ──────────────────────────────────────────────────────────────────────────────
# Genome index: token space + family tables for a set of species
# ──────────────────────────────────────────────────────────────────────────────

class GenomeIndex:
    """Token space over one or more species.

    Token ``t`` is row ``t`` of the stacked gene tables (species in the order
    given). Species ``s`` is referred to by its position in ``species``.

    Attributes (numpy):
      esm            [V, 2560] float16 L2-normalized protein embeddings
      fine_of_tok    [V] family of each token
      species_of_tok [V]
      fam_cnt[s, f]  number of genes of species s in family f
      fam_ptr[s, f]  start of the (s, f) member block in fam_mem
      fam_mem        member tokens grouped by (species, family), token order within a block
      rank_in_fam[t] position of token t inside its member block
      banks[s]       sorted families present in the genome of species s
    """

    def __init__(self, tables: Sequence[GeneTable], space: FamilySpace):
        for t in tables:
            if t.family is None:
                raise ValueError(f"{t.species}: families not assigned (use FamilySpace.assign)")
        self.species = [t.species for t in tables]
        self.space = space
        kf = space.k_fine
        self.offsets = np.concatenate([[0], np.cumsum([len(t) for t in tables])]).astype(np.int64)
        self.esm = np.concatenate([np.asarray(t.esm, dtype=np.float16) for t in tables])
        self.fine_of_tok = np.concatenate([t.family for t in tables]).astype(np.int64)
        self.species_of_tok = np.repeat(np.arange(len(tables)), [len(t) for t in tables])
        S, V = len(tables), len(self.fine_of_tok)
        self.fam_cnt = np.zeros((S, kf), dtype=np.int64)
        self.fam_ptr = np.zeros((S, kf), dtype=np.int64)
        self.rank_in_fam = np.zeros(V, dtype=np.int64)
        mem_parts, off, self.banks = [], 0, []
        for s in range(S):
            toks = np.arange(self.offsets[s], self.offsets[s + 1], dtype=np.int64)
            f = self.fine_of_tok[toks]
            order = np.argsort(f, kind="stable")
            toks_s, f_s = toks[order], f[order]
            cnt = np.bincount(f_s, minlength=kf).astype(np.int64)
            starts = np.concatenate([[0], np.cumsum(cnt)])[:-1]
            self.fam_cnt[s] = cnt
            self.fam_ptr[s] = starts + off
            self.rank_in_fam[toks_s] = np.arange(len(toks_s)) - starts[f_s]
            mem_parts.append(toks_s)
            off += len(toks_s)
            self.banks.append(np.flatnonzero(cnt > 0).astype(np.int64))
        self.fam_mem = np.concatenate(mem_parts)
        self.gene_ids = [t.gene_ids for t in tables]

    @property
    def n_tokens(self) -> int:
        return int(len(self.fine_of_tok))

    def tokens_of(self, species: str, names) -> np.ndarray:
        """Global token ids of raw gene names of ``species`` (-1 if absent)."""
        s = self.species.index(species)
        tab = GeneTable(species, self.gene_ids[s], self.esm[self.offsets[s]:self.offsets[s + 1]])
        loc = tab.map_genes(names)
        return np.where(loc >= 0, loc + self.offsets[s], -1)

    def family_sampler(self, n_samples: int) -> StratifiedSampler:
        return StratifiedSampler(self.banks, self.space.coarse_of_fine, self.space.k_coarse, n_samples)


class GenomeMeasure:
    """Support of the encoder's predicted family measure for each species.

    Training: stratified sample of ``n_samples`` families (log-weights log(n_j/k_j)).
    Evaluation: every family present in the genome, exactly (log-weights 0).
    Not an ``nn.Module``: nothing here is saved in checkpoints.
    """

    def __init__(self, index: GenomeIndex, n_samples: int, device):
        banks = index.banks
        S, Fmax = len(banks), max(len(b) for b in banks)
        full = np.zeros((S, Fmax), dtype=np.int64)
        msk = np.zeros((S, Fmax), dtype=bool)
        for s, b in enumerate(banks):
            full[s, :len(b)] = b
            msk[s, :len(b)] = True
        self.full_fid = torch.from_numpy(full).to(device)
        self.full_mask = torch.from_numpy(msk).to(device)
        self.sampler = index.family_sampler(n_samples).to(device)

    def restrict(self, species: int, families: np.ndarray) -> None:
        """Evaluation only: use ``families`` (sorted, unique) as the support of the
        predicted measure of ``species`` instead of all families of its genome."""
        fam = torch.as_tensor(np.unique(np.asarray(families, dtype=np.int64)), device=self.full_fid.device)
        S, Fmax = self.full_fid.shape
        W = len(fam) if S == 1 else max(Fmax, len(fam))
        fid = torch.zeros(S, W, dtype=torch.long, device=fam.device)
        msk = torch.zeros(S, W, dtype=torch.bool, device=fam.device)
        n = min(Fmax, W)
        fid[:, :n], msk[:, :n] = self.full_fid[:, :n], self.full_mask[:, :n]
        fid[species], msk[species] = 0, False
        fid[species, :len(fam)], msk[species, :len(fam)] = fam, True
        self.full_fid, self.full_mask = fid, msk

    def get(self, species: Tensor, training: bool,
            generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
        """species: [S] genome ids -> (family ids [S, K], log weights [S, K])."""
        if training:
            fid, w = self.sampler.sample(species, generator=generator)
            return fid, torch.log(w.clamp(min=1e-30))
        fid = self.full_fid[species]
        lw = torch.zeros(fid.shape, device=fid.device, dtype=torch.float32)
        return fid, lw.masked_fill(~self.full_mask[species], float("-inf"))
