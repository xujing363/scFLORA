"""Demo: embed 48 synthetic Arabidopsis cells and predict genome-wide expression.

    python examples/demo.py --assets assets/ [--device cpu]

The cells and the expected outputs (computed with the code used in the paper) are
in tests/data/reference_arabidopsis.npz.
"""
import argparse
import os
import time

import numpy as np
import scipy.sparse as sp
import torch

import scflora
from scflora.data import MatrixPoints, PointCollator

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    t0 = time.time()
    model = scflora.load_pretrained(f"{a.assets}/scflora_v1.pt", f"{a.assets}/family_space.npz", device=a.device)
    table = scflora.prepare_species(model, f"{a.assets}/genes/arabidopsis.npz")
    print(f"model loaded; Arabidopsis gene table: {len(table):,} genes ({time.time() - t0:.1f}s)")

    z = np.load(os.path.join(HERE, "..", "tests", "data", "reference_arabidopsis.npz"))
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    var_names = z["var_names"]
    print(f"input: {X.shape[0]} cells x {X.shape[1]} genes (raw counts)")

    t1 = time.time()
    E = scflora.embed(model, X, var_names, amp=False, num_workers=0)
    print(f"embeddings: {E.shape} ({time.time() - t1:.2f}s)")

    # genome-wide predicted rates of the first 4 cells (encoded together with the others,
    # as in the reference)
    col2tok = model.genome.tokens_of("arabidopsis", var_names)
    b = PointCollator()([MatrixPoints(X, col2tok)[i] for i in range(X.shape[0])]).to(a.device)
    with torch.no_grad():
        o = model(b)
        ll, gtok = model.log_rates(o["z"][:4], o["depth_eff"][:4])
    rates = torch.exp(ll).cpu().numpy()
    print(f"predicted rates: {rates.shape}; total per cell = {rates.sum(1).round(1)} "
          f"(observed depth {np.asarray(X[:4].sum(1)).ravel()})")

    gpu = model.log_beta.is_cuda
    ok = (np.abs(E - z["embedding"]).max() < (1e-5 if gpu else 5e-3)
          and np.abs(ll.cpu().numpy() - z["log_rates"]).max() < (1e-4 if gpu else 2e-2))
    print(f"matches reference: {ok}   (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
