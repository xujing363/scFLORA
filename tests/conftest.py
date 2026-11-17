"""Toy assets: two species, 12 families in 3 strata, 16-d 'protein embeddings'."""
import numpy as np
import pytest
import torch

from scflora import FamilySpace, GeneTable, GenomeIndex, ModelConfig, PointBatch, ScFLORA
from scflora.genes import normalize_esm

D_E, K_FINE, K_COARSE = 16, 12, 3


def toy_space(seed=0):
    rng = np.random.default_rng(seed)
    C = normalize_esm(rng.normal(size=(K_FINE, D_E)))
    return FamilySpace(C, np.arange(K_FINE) % K_COARSE, K_COARSE)


def toy_table(name, n, space, seed):
    rng = np.random.default_rng(seed)
    esm = normalize_esm(rng.normal(size=(n, D_E)))
    t = GeneTable(name, np.array([f"{name.upper()}{i:04d}" for i in range(n)]), esm)
    t.family = space.assign(t.esm)
    return t


def toy_config():
    return ModelConfig(esm_dim=D_E, d_model=16, n_heads=2, d_ff=32, n_latents=4, n_summary=2,
                       n_enc_layers=2, n_dec_layers=1, n_dec_slots=2, dropout=0.0, rma_samples=8,
                       member_chunk=4)


@pytest.fixture
def toy():
    torch.manual_seed(0)
    space = toy_space()
    tables = [toy_table("spa", 60, space, 1), toy_table("spb", 45, space, 2)]
    index = GenomeIndex(tables, space)
    model = ScFLORA(toy_config(), space).double().eval()
    model.set_genome(index, device="cpu")
    model.double()
    return model, index


def toy_batch(index, seed=0, lengths=(8, 6, 10), species=(0, 1, 0), dtype=torch.float64):
    rng = np.random.default_rng(seed)
    B, N = len(lengths), max(lengths)
    tok = torch.zeros(B, N, dtype=torch.long)
    cnt = torch.zeros(B, N, dtype=dtype)
    valid = torch.zeros(B, N, dtype=torch.bool)
    for i, (n, s) in enumerate(zip(lengths, species)):
        t = rng.choice(np.arange(index.offsets[s], index.offsets[s + 1]), n, replace=False)
        tok[i, :n] = torch.from_numpy(t)
        cnt[i, :n] = torch.from_numpy(rng.integers(1, 20, size=n).astype(np.float64))
        valid[i, :n] = True
    return PointBatch(tok=tok, cnt=cnt, valid=valid, species=torch.tensor(species), depth=cnt.sum(1))
