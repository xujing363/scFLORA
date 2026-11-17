# scFLORA

**A reproducible generative model for genome-wide perturbation analysis across plant species.**

scFLORA maps raw plant single-cell RNA-seq counts to a 512-dimensional cell state using protein-informed gene representations, residual-measure attention, and a family-factorized Poisson decoder. The released checkpoint is the exact model used for inference at training step 200,000.

## Reproducible release

This repository contains the source code, configuration, tests, examples, and the 36 MB release checkpoint. Large frozen species assets are downloaded separately and verified by SHA-256. The release is deterministic under fixed software, device, seed, and input files.

```bash
git clone https://github.com/xujing363/scFLORA.git
cd scFLORA
conda env create -f environment.yml
conda activate scflora
pip install -e .
sha256sum -c scflora_v1.pt.sha256
pytest tests
```

The checkpoint is `scflora_v1.pt`. Download `family_space.npz` and the required `genes/<species>.npz` from the release asset archive described in `docs/assets.md`, place them under `assets/`, and verify `docs/SHA256SUMS.txt` before analysis. Never substitute a different family space or gene table.

## Workflow overview

| Downstream task | Input | Command/API | Output |
|---|---|---|---|
| Cell embedding, annotation, integration | raw-count `.h5ad` | `scflora embed` | `obsm["X_scflora"]`, 512 dimensions |
| Genome-wide expression prediction | raw counts plus species gene table | `scflora predict` or `predict_rates` | expected rates for every annotated gene |
| Virtual gene knockout | wild-type raw counts and target genes | `scflora knockout` or `virtual_knockout` | response and allocation scores |
| New species transfer | longest-isoform protein FASTA plus counts | `scflora protein-embed`, then `scflora embed` | species gene table and embeddings |
| Pretraining from scratch | manifest, packed corpus, gene tables | `scflora pack`, `scflora train` | checkpoint and training manifest |

All commands below assume `assets/` contains the verified family space and species tables.

## 1. Cell embeddings and cross-species integration

Input must be raw counts with cells in rows and gene identifiers matching the species annotation. Gene order is handled by the package. Use the genome support for full-genome inference and input support for a restricted HVG panel.

```bash
scflora embed \
  --checkpoint scflora_v1.pt \
  --family-space assets/family_space.npz \
  --genes assets/genes/arabidopsis.npz \
  --h5ad data/arabidopsis_root.h5ad \
  --out results/arabidopsis_scflora.h5ad \
  --support genome
```

For an integration benchmark, select HVGs independently within each training split, pass the same panel definition to every method, and use `--support input`. Keep patient or biological replicate splits outside the embedding step to prevent leakage.

## 2. Genome-wide predicted expression

The decoder returns expected counts at the observed sequencing depth, including genes absent from the measured matrix.

```python
import anndata as ad
import scflora

model = scflora.load_pretrained("scflora_v1.pt", "assets/family_space.npz", device="cuda:0")
scflora.prepare_species(model, "assets/genes/arabidopsis.npz")
adata = ad.read_h5ad("data/arabidopsis_root.h5ad")
out = scflora.predict_rates(model, adata.X, adata.var_names)
# out["genes"]: gene identifiers; out["rates"]: cells x genes expected counts
```

Save the input checksum, package version, command line, device, and output checksum with every run.

## 3. Virtual gene knockout

For each target, only the target counts are set to zero. Wild-type cells, depth, model weights, and species gene assignments remain fixed.

```python
result = scflora.virtual_knockout(
    model, adata.X, adata.var_names, targets=["AT4G37650"]
)
result["score"]
result["log2_response"]
```

Use a prespecified target list and report the number of cells, target coverage, and whether targets were expressed in the wild-type input.

## 4. Transfer to a species absent from pretraining

Use one longest protein isoform per gene and retain the FASTA headers needed to match the count matrix.

```bash
scflora protein-embed \
  --fasta data/Setaria_italica.pep.fa.gz \
  --out assets/genes/Setaria_italica.npz
scflora embed \
  --checkpoint scflora_v1.pt \
  --family-space assets/family_space.npz \
  --genes assets/genes/Setaria_italica.npz \
  --h5ad data/setaria.h5ad \
  --out results/setaria_scflora.h5ad
```

Unseen-species genes are assigned to the nearest released family centroid. Record the FASTA checksum and the exact ESM-2 model revision.

## 5. Reproduce pretraining

The released configuration fixes the optimizer, schedule, precision, masking distribution, sampler, and epoch count. The corpus manifest and packed files must be generated from the same source data.

```bash
scflora pack --manifest corpus.tsv --gene-dir assets/genes \
  --species species.txt --family-space assets/family_space.npz \
  --cache-dir run/cache --pack-dir run/pack
scflora train --config configs/scflora_v1_pretrain.json \
  --manifest corpus.tsv --gene-dir assets/genes --species species.txt \
  --family-space assets/family_space.npz --pack-dir run/pack \
  --out-dir run --devices 0
```

The released run uses AdamW, learning rate 3e-4 with 2% warm-up and cosine decay to 1e-5, bfloat16, visible fraction sampled from U(0.3, 0.9), 4,096 sampled families, and 15 epochs. Resume only from a checkpoint whose configuration and manifest hashes match.

## Verification and provenance

```bash
pytest tests
python examples/demo.py --assets assets/
```

The demo must report a `(48, 512)` embedding, predicted rates, and `matches reference: True`. Keep `environment.yml`, all input checksums, command lines, random seeds, GPU model, and output checksums with the result. Detailed artifact requirements and the per-analysis support settings are in [`docs/reproducibility.md`](docs/reproducibility.md). Asset formats and checksums are in [`docs/assets.md`](docs/assets.md).

## Citation and license

See [`CITATION.cff`](CITATION.cff) and [`LICENSE`](LICENSE). Please open an issue with the complete command, commit hash, asset checksums, and error log when reporting a reproducibility problem.
