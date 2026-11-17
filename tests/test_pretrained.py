"""Regression test against reference outputs of the code used for the paper.

Requires the released assets; set SCFLORA_ASSETS to the directory holding
scflora_v1.pt, family_space.npz and genes/arabidopsis.npz. Skipped otherwise.
"""
import os

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import scflora

ASSETS = os.environ.get("SCFLORA_ASSETS")
REF = os.path.join(os.path.dirname(__file__), "data", "reference_arabidopsis.npz")
pytestmark = pytest.mark.skipif(not ASSETS, reason="set SCFLORA_ASSETS to run")


@pytest.fixture(scope="module")
def model_and_ref():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    m = scflora.load_pretrained(f"{ASSETS}/scflora_v1.pt", f"{ASSETS}/family_space.npz", device=dev)
    scflora.prepare_species(m, f"{ASSETS}/genes/arabidopsis.npz")
    z = np.load(REF)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return m, z, X


def test_embeddings_match_reference(model_and_ref):
    m, z, X = model_and_ref
    E = scflora.embed(m, X, z["var_names"], amp=False, num_workers=0)
    tol = 1e-5 if m.log_beta.is_cuda else 5e-3          # reference computed on GPU (TF32)
    assert np.abs(E - z["embedding"]).max() < tol


def test_log_rates_match_reference(model_and_ref):
    m, z, X = model_and_ref
    from scflora.data import MatrixPoints, PointCollator
    col2tok = m.genome.tokens_of("arabidopsis", z["var_names"])
    b = PointCollator()([MatrixPoints(X, col2tok)[i] for i in range(X.shape[0])]).to(m.log_beta.device)
    with torch.no_grad():
        o = m(b)                                         # same batch as the reference
        ll, gtok = m.log_rates(o["z"][:4], o["depth_eff"][:4])
    assert np.array_equal(m.genome.gene_ids[0][gtok.cpu().numpy()], z["rate_genes"])
    tol = 1e-4 if m.log_beta.is_cuda else 2e-2
    assert np.abs(ll.cpu().numpy() - z["log_rates"]).max() < tol
