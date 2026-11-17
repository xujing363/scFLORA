"""Virtual knockout on wild-type cells.

    python examples/virtual_knockout.py --assets assets/ --species arabidopsis \
        --h5ad wt_root_raw_counts.h5ad --targets AT4G37650 --out shr_ko

Writes <out>_ranking.tsv (candidate genes ranked by the mean absolute change in the
decoder's within-family allocation logit) and <out>_response.tsv (signed genome-wide
predicted response, log2 sum KO / sum WT at wild-type depth).
"""
import argparse

import anndata as ad
import pandas as pd

import scflora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    model = scflora.load_pretrained(f"{a.assets}/scflora_v1.pt", f"{a.assets}/family_space.npz", device=a.device)
    scflora.prepare_species(model, f"{a.assets}/genes/{a.species}.npz")
    wt = ad.read_h5ad(a.h5ad)
    r = scflora.virtual_knockout(model, wt.X, wt.var_names, a.targets)
    pd.DataFrame({"gene": r["candidates"], "score": r["score"]}).sort_values("score", ascending=False) \
        .to_csv(f"{a.out}_ranking.tsv", sep="\t", index=False)
    pd.DataFrame({"gene": r["genes"], "log2_response": r["log2_response"]}) \
        .to_csv(f"{a.out}_response.tsv", sep="\t", index=False)
    print(f"{r['n_cells']:,} wild-type cells; wrote {a.out}_ranking.tsv and {a.out}_response.tsv")


if __name__ == "__main__":
    main()
