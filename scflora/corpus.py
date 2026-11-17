"""Pretraining corpus: index, cell filters, train/validation split, packing and
species-balanced length-bucketed sampling.

The corpus is described by a manifest (TSV) with columns
``dataset_key``, ``species`` and ``path`` (an .h5ad whose ``X`` is a CSR matrix
of raw counts). Genes are mapped to tokens of a ``GenomeIndex`` built from the
gene tables of all pretraining species (``col2tok``). Cells are filtered
(>= ``min_expressed_genes`` genes in the gene tables; total counts within
+/- ``outlier_sd`` standard deviations of the dataset mean), and 1% of the
eligible cells of every dataset (capped) are held out for validation.

For speed, the filtered corpus is packed once into three flat memory-mapped
arrays (tokens, counts, row pointers).
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch

from .config import DataConfig
from .families import GenomeIndex


def _var_names(f: h5py.File) -> np.ndarray:
    var = f["var"]
    key = var.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    v = var[key]
    if isinstance(v, h5py.Group):                       # categorical encoding
        cats = v["categories"][:]
        v = cats[v["codes"][:]]
    else:
        v = v[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])


class Corpus:
    def __init__(self, manifest: str, index: GenomeIndex, cache_dir: str, dcfg: DataConfig,
                 seed: int = 42, col2tok_dir: Optional[str] = None, verbose: bool = True):
        with open(manifest, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f, delimiter="\t"))
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir, self.index = cache_dir, index
        self.ds_key = [r["dataset_key"] for r in self.rows]
        self.paths = [r["path"] for r in self.rows]
        self.species_of_ds = np.array([index.species.index(r["species"]) for r in self.rows], dtype=np.int64)
        self.col2tok: List[np.ndarray] = []
        self.indptr: List[np.ndarray] = []
        for r in self.rows:
            with h5py.File(r["path"], "r") as f:
                ip = np.asarray(f["X"]["indptr"][:], dtype=np.int64)
                pre = None if col2tok_dir is None else os.path.join(col2tok_dir, f"{r['dataset_key']}.npy")
                if pre is not None and os.path.exists(pre):
                    c2t = np.load(pre)
                else:
                    c2t = index.tokens_of(r["species"], _var_names(f))
            if (c2t >= 0).sum() == 0:
                raise ValueError(f"{r['dataset_key']}: no gene maps to the {r['species']} gene table")
            self.col2tok.append(c2t.astype(np.int64))
            self.indptr.append(ip)
            if verbose:
                print(f"[corpus] {r['dataset_key']}: {len(ip) - 1:,} cells, "
                      f"{int((c2t >= 0).sum()):,}/{len(c2t):,} genes in table", flush=True)
        self.n_cells = [len(ip) - 1 for ip in self.indptr]
        self.offsets = np.concatenate([[0], np.cumsum(self.n_cells)]).astype(np.int64)
        self.total_cells = int(self.offsets[-1])
        self.ds_of_cell = np.concatenate([np.full(n, k, dtype=np.int32) for k, n in enumerate(self.n_cells)])
        self.nnz, self.depth = self._scan()
        self._split(dcfg, seed, verbose)

    def _scan(self):
        """Per cell: number of in-table positive genes and in-table total counts (cached)."""
        nnz_all, dep_all = [], []
        for k, key in enumerate(self.ds_key):
            pn = os.path.join(self.cache_dir, f"nnz__{key}.npy")
            pd_ = os.path.join(self.cache_dir, f"depth__{key}.npy")
            if os.path.exists(pn) and os.path.exists(pd_):
                nnz_all.append(np.load(pn)); dep_all.append(np.load(pd_)); continue
            ip, ok, n = self.indptr[k], self.col2tok[k] >= 0, self.n_cells[k]
            cnt, dep = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float64)
            with h5py.File(self.paths[k], "r") as f:
                I, D = f["X"]["indices"], f["X"]["data"]
                for a in range(0, n, 100_000):
                    b = min(a + 100_000, n)
                    s0, e0 = int(ip[a]), int(ip[b])
                    if e0 <= s0:
                        continue
                    cols = np.asarray(I[s0:e0], dtype=np.int64)
                    vals = np.asarray(D[s0:e0], dtype=np.float64)
                    hit = ok[cols]
                    vals[~hit] = 0.0
                    lip = ip[a:b + 1] - s0
                    ch = np.concatenate([[0], np.cumsum(hit)])
                    cv = np.concatenate([[0.0], np.cumsum(vals)])
                    cnt[a:b] = ch[lip[1:]] - ch[lip[:-1]]
                    dep[a:b] = cv[lip[1:]] - cv[lip[:-1]]
            np.save(pn, cnt); np.save(pd_, dep)
            nnz_all.append(cnt); dep_all.append(dep)
        return np.concatenate(nnz_all), np.concatenate(dep_all)

    def _split(self, dcfg: DataConfig, seed: int, verbose: bool) -> None:
        rng = np.random.default_rng(12345 + seed)
        eligible = self.nnz >= max(dcfg.min_expressed_genes, 1)
        for k in range(len(self.ds_key)):
            a, b = int(self.offsets[k]), int(self.offsets[k + 1])
            m, d = eligible[a:b], self.depth[a:b]
            if m.sum() < 2 or dcfg.outlier_sd <= 0:
                continue
            mu, sd = d[m].mean(), d[m].std()
            eligible[a:b] = m & (d > mu - dcfg.outlier_sd * sd) & (d < mu + dcfg.outlier_sd * sd)
        val = []
        for k in range(len(self.ds_key)):
            a, b = int(self.offsets[k]), int(self.offsets[k + 1])
            ok = np.flatnonzero(eligible[a:b]) + a
            if len(ok) == 0:
                continue
            m = max(1, int(len(ok) * dcfg.val_frac))
            if dcfg.val_cap > 0:
                m = min(m, max(1, dcfg.val_cap // len(self.ds_key)))
            val.append(rng.choice(ok, size=min(m, len(ok)), replace=False))
        self.val_idx = np.sort(np.concatenate(val))
        mask = eligible.copy()
        mask[self.val_idx] = False
        self.train_idx = np.flatnonzero(mask)
        if verbose:
            print(f"[corpus] {self.total_cells:,} cells; eligible {int(eligible.sum()):,}; "
                  f"train {len(self.train_idx):,} / val {len(self.val_idx):,}", flush=True)

    def exclude(self, mask_npz: str) -> None:
        """Remove cells flagged in a boolean ``mask`` over the global cell index
        (e.g. cells overlapping evaluation datasets)."""
        m = np.load(mask_npz)["mask"].astype(bool)
        if len(m) != self.total_cells:
            raise ValueError("exclusion mask length does not match the corpus")
        self.train_idx = self.train_idx[~m[self.train_idx]]
        self.val_idx = self.val_idx[~m[self.val_idx]]

    def sample_weights(self, idx: np.ndarray, alpha: float) -> np.ndarray:
        """p(species s) proportional to n_s ** alpha, uniform within species."""
        sp = self.species_of_ds[self.ds_of_cell[idx]]
        n_s = np.bincount(sp, minlength=len(self.index.species)).astype(np.float64)
        n_s[n_s == 0] = 1.0
        p_s = n_s ** alpha
        p_s = p_s / p_s.sum()
        return (p_s / n_s)[sp]

    # ── packing ───────────────────────────────────────────────────────────────
    def pack(self, pack_dir: str) -> None:
        """Write tok.u32 / cnt.f32 / ptr.i64 for all cells (in-table genes only,
        duplicate gene columns merged)."""
        os.makedirs(pack_dir, exist_ok=True)
        ptr = np.zeros(self.total_cells + 1, dtype=np.int64)
        ft, fc = open(os.path.join(pack_dir, "tok.u32"), "wb"), open(os.path.join(pack_dir, "cnt.f32"), "wb")
        g = 0
        for k in range(len(self.ds_key)):
            c2t, ip, ok = self.col2tok[k], self.indptr[k], self.col2tok[k] >= 0
            with h5py.File(self.paths[k], "r") as f:
                I, D = f["X"]["indices"], f["X"]["data"]
                for a in range(0, self.n_cells[k], 20_000):
                    b = min(a + 20_000, self.n_cells[k])
                    s0, e0 = int(ip[a]), int(ip[b])
                    cols = np.asarray(I[s0:e0], dtype=np.int64)
                    vals = np.asarray(D[s0:e0], dtype=np.float32)
                    lip = ip[a:b + 1] - s0
                    for r in range(b - a):
                        lo, hi = int(lip[r]), int(lip[r + 1])
                        c = cols[lo:hi]; v = vals[lo:hi]
                        keep = ok[c]
                        t, v = c2t[c[keep]], v[keep]
                        if len(t) > 1:
                            u, inv = np.unique(t, return_inverse=True)
                            if len(u) != len(t):
                                mv = np.zeros(len(u), dtype=np.float32)
                                np.add.at(mv, inv, v)
                                t, v = u, mv
                        ft.write(t.astype(np.uint32).tobytes()); fc.write(v.astype(np.float32).tobytes())
                        ptr[g + 1] = ptr[g] + len(t)
                        g += 1
            print(f"[pack] {self.ds_key[k]} done", flush=True)
        ft.close(); fc.close()
        assert g == self.total_cells
        ptr.tofile(os.path.join(pack_dir, "ptr.i64"))


class PackedCells(torch.utils.data.Dataset):
    def __init__(self, corpus: Corpus, pack_dir: str, idx: np.ndarray):
        self.ptr = np.fromfile(os.path.join(pack_dir, "ptr.i64"), dtype=np.int64)
        if len(self.ptr) != corpus.total_cells + 1:
            raise RuntimeError("packed corpus does not match the manifest; re-run `scflora pack`")
        self.pack_dir, self.idx = pack_dir, np.asarray(idx, dtype=np.int64)
        self.species_of_cell = corpus.species_of_ds[corpus.ds_of_cell]
        self._tok = self._cnt = None

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict:
        if self._tok is None:                          # one memmap per worker
            self._tok = np.memmap(os.path.join(self.pack_dir, "tok.u32"), dtype=np.uint32, mode="r")
            self._cnt = np.memmap(os.path.join(self.pack_dir, "cnt.f32"), dtype=np.float32, mode="r")
        g = int(self.idx[i])
        a, b = int(self.ptr[g]), int(self.ptr[g + 1])
        return {"tok": np.asarray(self._tok[a:b], dtype=np.int64),
                "val": np.asarray(self._cnt[a:b], dtype=np.float32),
                "species": int(self.species_of_cell[g])}


class BucketedWeightedBatchSampler(torch.utils.data.Sampler):
    """Each epoch draws ``n_per_epoch`` cells with replacement according to
    ``weights``; draws are grouped into megabatches, sorted by length and cut into
    batches under a token budget (cells x longest cell); batch order is shuffled.
    Bucketing changes only which cells share a batch, not the sampling distribution.
    With DDP all ranks build the same batch list and take every world-th batch."""

    def __init__(self, weights, lengths, batch_size: int, n_per_epoch: int, bucket_mult: int = 64,
                 seed: int = 42, max_tokens: int = 0, max_batch: int = 256, rank: int = 0, world: int = 1):
        self.w = torch.as_tensor(weights, dtype=torch.double)
        self.lengths = np.asarray(lengths)
        self.bs, self.n, self.mult, self.seed = int(batch_size), int(n_per_epoch), max(1, int(bucket_mult)), int(seed)
        self.max_tokens, self.max_batch = int(max_tokens), int(max_batch)
        self.rank, self.world = int(rank), max(1, int(world))
        self.epoch, self.skip = 0, 0
        self._cache: Dict[int, List[np.ndarray]] = {}

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def _build(self, epoch: int) -> List[np.ndarray]:
        if epoch in self._cache:
            return self._cache[epoch]
        g = torch.Generator().manual_seed(self.seed + 1000 * epoch)
        idx = torch.multinomial(self.w, self.n, replacement=True, generator=g).numpy()
        mb, batches = self.bs * self.mult, []
        for a in range(0, len(idx), mb):
            chunk = idx[a:a + mb]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            if self.max_tokens > 0:
                cur: List[int] = []
                for c in chunk:
                    L = int(self.lengths[c])
                    if cur and ((len(cur) + 1) * L > self.max_tokens or len(cur) >= self.max_batch):
                        batches.append(np.asarray(cur)); cur = []
                    cur.append(int(c))
                if cur:
                    batches.append(np.asarray(cur))
            else:
                batches += [chunk[b:b + self.bs] for b in range(0, len(chunk), self.bs)
                            if len(chunk[b:b + self.bs]) == self.bs]
        if self.world > 1:
            ng = len(batches) // self.world
            batches = batches[:ng * self.world]
            out = [batches[i * self.world + self.rank] for i in torch.randperm(ng, generator=g).tolist()]
        else:
            out = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
        self._cache = {epoch: out}
        return out

    def __len__(self) -> int:
        return len(self._build(self.epoch))

    def __iter__(self):
        bl = self._build(self.epoch)
        k, self.skip = int(self.skip), 0
        for b in bl[k:]:
            yield b.tolist()
