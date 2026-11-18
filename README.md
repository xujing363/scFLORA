# scFLORA

## A checkpoint-first resource for plant single-cell perturbation inference

scFLORA is a protein-informed generative model for plant single-cell transcriptomes. This repository is limited to the frozen-checkpoint inference layer used for downstream analyses. It provides auditable commands for cell-state embedding, cross-species transfer, genome-wide expression prediction, virtual knockout scoring, and new-species preparation.

### Release contents

- `scflora_v1.pt` — released checkpoint, step 200,000.
- `scflora/` — inference implementation and CLI.
- `examples/` — minimal executable workflows.
- `tests/` — numerical and regression checks.
- `docs/assets.md` — asset schema and SHA-256 manifest.
- `docs/reproducibility.md` — analysis contract and reporting checklist.

Training code, pretraining data, internal audit logs, and manuscript-specific pipelines are outside this inference release. They are not required to reproduce checkpoint-based analyses.

## Quick start

```bash
git clone https://github.com/xujing363/scFLORA.git
cd scFLORA
conda env create -f environment.yml
conda activate scflora
pip install -e .
sha256sum -c scflora_v1.pt.sha256
pytest tests -q
```

Download the required `family_space.npz` and species gene table from the versioned release archive. Verify every file against `docs/SHA256SUMS.txt` and place them under `assets/`.

## Supported downstream analyses

### Cell-state embedding

```bash
scflora embed --checkpoint scflora_v1.pt --family-space assets/family_space.npz \
  --genes assets/genes/arabidopsis.npz --h5ad data/arabidopsis.h5ad \
  --out results/arabidopsis_scflora.h5ad --support genome
```

The output stores the 512-dimensional representation in `obsm["X_scflora"]`. For a predeclared HVG panel, add `--hvg N --support input`. HVG selection must be performed within each training split.

### Genome-wide predicted expression

```bash
scflora predict --checkpoint scflora_v1.pt --family-space assets/family_space.npz \
  --genes assets/genes/arabidopsis.npz --h5ad data/arabidopsis.h5ad \
  --out results/arabidopsis_rates.npz
```

The output contains `genes`, `rates` and `obs_names`. Rates are expected counts at each cell's observed sequencing depth, including unmeasured annotated genes.

### Virtual knockout

```bash
scflora knockout --checkpoint scflora_v1.pt --family-space assets/family_space.npz \
  --genes assets/genes/arabidopsis.npz --h5ad data/arabidopsis_wildtype.h5ad \
  --targets AT4G37650 AT1G01010 --out results/ko
```

This writes ranked allocation scores and gene-level log2 responses. Report target list, wild-type cell count, target detection rate, and asset checksums.

### New-species transfer

```bash
scflora protein-embed --fasta data/species.pep.fa.gz --out assets/genes/species.npz
scflora embed --checkpoint scflora_v1.pt --family-space assets/family_space.npz \
  --genes assets/genes/species.npz --species species --h5ad data/species.h5ad \
  --out results/species_scflora.h5ad
```

Use one longest protein isoform per gene and retain the FASTA checksum. New-species genes are assigned to the nearest released family centroid.

## Reproducibility standard

For every result archive the repository commit, command line, environment lock, input checksums, checkpoint, family-space and species-table checksums, random seed, device, precision, cell count, gene count, and output checksum. Run `pytest tests -q` and the demo before biological analysis. The complete reporting contract is in [`docs/reproducibility.md`](docs/reproducibility.md).

## Citation

See [`CITATION.cff`](CITATION.cff). Code is released under the MIT license.
