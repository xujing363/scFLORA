"""Command-line interface: ``scflora <command> --help``.

  protein-embed   ESM-2 embeddings of a proteome FASTA (new species)
  embed           cell embeddings of an .h5ad with raw counts
  predict         genome-wide predicted expression for cells of an .h5ad
  knockout        virtual knockout of one or more genes on wild-type cells
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _threads(n: int) -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, str(n))
    import torch
    torch.set_num_threads(n)


def _load(args):
    import scflora
    model = scflora.load_pretrained(args.checkpoint, args.family_space, device=args.device)
    if args.genes.endswith(".npz") and "family" in np.load(args.genes).files:
        scflora.prepare_species(model, args.genes)
    else:
        if not args.species:
            sys.exit("--species is required with a new ESM-2 embedding file")
        scflora.prepare_species(model, args.genes, species=args.species)
    return model


def _read(args):
    import anndata as ad
    a = ad.read_h5ad(args.h5ad)
    X = a.layers[args.layer] if args.layer else a.X
    return a, X


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="scflora")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--checkpoint", required=True, help="scflora_v1.pt")
        q.add_argument("--family-space", required=True, help="family_space.npz")
        q.add_argument("--genes", required=True,
                       help="released gene table genes/<species>.npz, or an ESM-2 file of a new species")
        q.add_argument("--species", default=None, help="species name (required for a new species)")
        q.add_argument("--h5ad", required=True, help="cells x genes, raw counts")
        q.add_argument("--layer", default=None, help="use adata.layers[LAYER] instead of X")
        q.add_argument("--device", default="cuda:0")
        q.add_argument("--threads", type=int, default=8)

    q = sub.add_parser("protein-embed", help="ESM-2 3B embeddings of a proteome")
    q.add_argument("--fasta", required=True)
    q.add_argument("--out", required=True, help="output .npz (gene_ids, emb)")
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--esm-model", default="facebook/esm2_t36_3B_UR50D",
                   help="HuggingFace model id or local directory of ESM-2 3B")

    q = sub.add_parser("embed", help="512-d cell embeddings")
    common(q)
    q.add_argument("--out", required=True, help="output .h5ad (obsm['X_scflora']) or .npy")
    q.add_argument("--hvg", type=int, default=0, help="restrict input to N highly variable genes (0 = all genes)")
    q.add_argument("--batch-size", type=int, default=64)
    q.add_argument("--fp32", action="store_true", help="disable bfloat16 autocast")
    q.add_argument("--support", choices=["genome", "input"], default="genome",
                   help="support of the predicted family measure (default: species genome)")

    q = sub.add_parser("predict", help="genome-wide predicted expression")
    common(q)
    q.add_argument("--out", required=True, help="output .npz (genes, rates)")

    q = sub.add_parser("knockout", help="virtual gene knockout on wild-type cells")
    common(q)
    q.add_argument("--targets", required=True, nargs="+", help="gene ids to knock out (all together)")
    q.add_argument("--candidates", default=None, help="file with candidate gene ids (default: all measured genes)")
    q.add_argument("--out", required=True, help="output prefix")

    a = p.parse_args(argv)

    if a.cmd == "protein-embed":
        from .protein import embed_proteome
        embed_proteome(a.fasta, a.out, device=a.device, model_name=a.esm_model)
        return

    if a.cmd in ("embed", "predict", "knockout"):
        _threads(a.threads)
        import scflora
        model = _load(a)
        adata, X = _read(a)
        names = adata.var_names.astype(str)
        if a.cmd == "embed":
            genes = None
            if a.hvg:
                import scanpy as sc
                t = adata.copy()
                if a.layer:
                    t.X = t.layers[a.layer]
                sc.pp.normalize_total(t, target_sum=1e4); sc.pp.log1p(t)
                sc.pp.highly_variable_genes(t, n_top_genes=a.hvg, flavor="seurat")
                genes = np.asarray(t.var_names[t.var["highly_variable"].to_numpy()])
            E = scflora.embed(model, X, names, batch_size=a.batch_size, amp=not a.fp32, genes=genes,
                              support=a.support)
            if a.out.endswith(".npy"):
                np.save(a.out, E)
            else:
                adata.obsm["X_scflora"] = E
                adata.write_h5ad(a.out)
        elif a.cmd == "predict":
            r = scflora.predict_rates(model, X, names)
            np.savez_compressed(a.out, genes=r["genes"], rates=r["rates"].astype(np.float32),
                                obs_names=np.asarray(adata.obs_names.astype(str)))
        else:
            cand = None if a.candidates is None else [l.strip() for l in open(a.candidates) if l.strip()]
            r = scflora.virtual_knockout(model, X, names, a.targets, candidates=cand)
            import pandas as pd
            pd.DataFrame({"gene": r["candidates"], "score": r["score"]}).sort_values(
                "score", ascending=False).to_csv(a.out + "_ranking.tsv", sep="\t", index=False)
            pd.DataFrame({"gene": r["genes"], "log2_response": r["log2_response"]}).to_csv(
                a.out + "_response.tsv", sep="\t", index=False)
        return



if __name__ == "__main__":
    main()
