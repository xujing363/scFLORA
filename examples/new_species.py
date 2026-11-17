"""Embed cells of a species that was not in pretraining.

    python examples/new_species.py --assets assets/ --fasta proteome.pep.fa.gz \
        --species Setaria_italica --h5ad setaria_raw_counts.h5ad --out setaria_scflora.h5ad

1. ESM-2 3B embedding of the proteome (longest isoform per gene; needs a GPU with >= 16 GB).
2. Genes are assigned to the nearest protein-semantic family; nothing is trained.
3. Cells are embedded with the frozen model.
"""
import argparse
import os

import anndata as ad

import scflora
from scflora.protein import embed_proteome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--h5ad", required=True, help="raw counts; var_names = gene ids of the proteome")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    esm_file = os.path.splitext(a.out)[0] + "_esm2.npz"
    if not os.path.exists(esm_file):
        embed_proteome(a.fasta, esm_file, device=a.device)
    model = scflora.load_pretrained(f"{a.assets}/scflora_v1.pt", f"{a.assets}/family_space.npz", device=a.device)
    table = scflora.prepare_species(model, esm_file, species=a.species)
    adata = ad.read_h5ad(a.h5ad)
    n_hit = int((table.map_genes(adata.var_names) >= 0).sum())
    print(f"{n_hit:,} of {adata.n_vars:,} genes of the matrix are in the proteome")
    scflora.embed_adata(model, adata)
    adata.write_h5ad(a.out)
    print("wrote", a.out, adata.obsm["X_scflora"].shape)


if __name__ == "__main__":
    main()
