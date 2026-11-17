"""Gene identifiers and per-species gene tables.

A *gene table* holds, for one species, the canonical gene identifiers, their
ESM-2 protein embeddings (2,560-d; stored L2-normalized in float16, the form
used by the model) and their protein-semantic family
(the index of the nearest family centroid in the shared family space).
Genes are identified only by these protein embeddings; scFLORA has no
gene-specific parameters, so a new species only needs a new gene table.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

_CANON_PREFIX = ("gene-", "gene:", "transcript:", "rna-", "cds-", "agis_", "osnip_", "os-")
_CANON_VERSION_RE = re.compile(r"\.\d+$")
_CANON_MULTI_US = re.compile(r"_+")


def canon_gene_id(g: str) -> str:
    """Normalize a plant gene identifier so that annotation variants map to one key.

    1. strip common GFF/FASTA prefixes (``gene-``, ``transcript:``, ...);
    2. strip a purely numeric version/isoform suffix (``AT1G01010.1`` -> ``AT1G01010``;
       ``Glyma.01G000100`` is kept because the suffix contains letters);
    3. upper-case, map ``-`` and ``.`` to ``_`` and collapse repeated ``_``.
    """
    g = str(g).strip()
    gl = g.lower()
    for p in _CANON_PREFIX:
        if gl.startswith(p):
            g = g[len(p):]
            break
    g = _CANON_VERSION_RE.sub("", g)
    g = g.upper().replace("-", "_").replace(".", "_")
    return _CANON_MULTI_US.sub("_", g)


@dataclass
class GeneTable:
    """Protein-derived representation of one species' protein-coding genes."""

    species: str
    gene_ids: np.ndarray            # [n] str, canonical ids (unique)
    esm: np.ndarray                 # [n, 2560] float16, L2-normalized mean-pooled ESM-2 3B embeddings
    family: Optional[np.ndarray] = None   # [n] int64 family index, or None before assignment

    def __post_init__(self) -> None:
        self.gene_ids = np.asarray(self.gene_ids).astype(str)
        if self.esm.ndim != 2 or self.esm.shape[0] != len(self.gene_ids):
            raise ValueError(f"{self.species}: esm {self.esm.shape} vs {len(self.gene_ids)} genes")
        if len(set(self.gene_ids.tolist())) != len(self.gene_ids):
            raise ValueError(f"{self.species}: duplicated canonical gene ids")
        if self.family is not None:
            self.family = np.asarray(self.family, dtype=np.int64)
            if self.family.shape != (len(self.gene_ids),):
                raise ValueError(f"{self.species}: family vector has wrong length")

    def __len__(self) -> int:
        return len(self.gene_ids)

    @property
    def index(self) -> Dict[str, int]:
        return {g: i for i, g in enumerate(self.gene_ids.tolist())}

    def map_genes(self, names: Iterable[str]) -> np.ndarray:
        """Map raw gene names (e.g. ``adata.var_names``) to row indices; -1 if absent."""
        idx = self.index
        return np.array([idx.get(canon_gene_id(g), -1) for g in names], dtype=np.int64)

    # -- I/O -----------------------------------------------------------------
    def save(self, path: str) -> None:
        kw = dict(species=np.array(self.species), gene_ids=self.gene_ids.astype("U"),
                  esm=self.esm.astype(np.float16))
        if self.family is not None:
            kw["family"] = self.family
        np.savez(path, **kw)

    @staticmethod
    def load(path: str) -> "GeneTable":
        z = np.load(path, allow_pickle=False)
        return GeneTable(species=str(z["species"]), gene_ids=z["gene_ids"],
                         esm=np.asarray(z["esm"]),
                         family=(np.asarray(z["family"]) if "family" in z.files else None))

    @staticmethod
    def from_esm_file(path: str, species: str) -> "GeneTable":
        """Build a table from an ESM-2 embedding file written by ``scflora protein-embed``
        (``.npz`` with ``gene_ids``/``emb``) or an ``.h5`` with the same two datasets.
        Ids are canonicalized; for duplicated canonical ids the first occurrence is kept."""
        if path.endswith(".h5") or path.endswith(".hdf5"):
            import h5py
            with h5py.File(path, "r") as f:
                gids = [g.decode() if isinstance(g, bytes) else str(g) for g in f["gene_ids"][:]]
                emb = np.asarray(f["emb"][:])
        else:
            z = np.load(path, allow_pickle=True)
            gids = [str(g) for g in z["gene_ids"]]
            emb = np.asarray(z["emb"])
        keep, seen, names = [], set(), []
        for i, g in enumerate(gids):
            c = canon_gene_id(g)
            if c in seen:
                continue
            seen.add(c); keep.append(i); names.append(c)
        return GeneTable(species=species, gene_ids=np.array(names),
                         esm=normalize_esm(emb[np.asarray(keep, dtype=np.int64)]))


def normalize_esm(emb: np.ndarray) -> np.ndarray:
    """Raw ESM-2 embeddings -> L2-normalized float16 (normalized in float32)."""
    import torch
    import torch.nn.functional as F
    E = torch.as_tensor(np.asarray(emb, dtype=np.float32))
    return F.normalize(E, dim=1).to(torch.float16).numpy()
