# Reproducibility

## Equivalence with the code used in the paper

The released package is a re-implementation of the original training and evaluation code, with the
same parameter names apart from a renaming map and the same numerics. It was checked against the
original code on real data using the same checkpoint and the same cells, on one H100 GPU with TF32.

| Check | Data | Max abs. difference |
|---|---|---|
| Cell embeddings, fp32 and bf16 | *A. thaliana* root (2,000 cells), soybean root (2,000 cells) | 0 (bit-identical) |
| Cell embeddings, species not in pretraining (families by nearest centroid) | *Lycopodium japonicum* (2,000 cells) | 0 |
| Species-wise HVG2000 panel with `support="input"`, bf16, batch 128 | *A. thaliana*, soybean | 0 |
| Genome-wide log predicted rates (27,628 genes × 16 cells) | *A. thaliana* | 3.8 × 10⁻⁶ |
| Virtual-knockout allocation score (2,000 candidate genes) | *A. thaliana* | 9.5 × 10⁻⁷ |
| Masked training loss, same random generator | *A. thaliana* | 1.6 × 10⁻⁵ (relative 3 × 10⁻⁶) |

`tests/test_pretrained.py` repeats a subset of these checks against reference outputs stored in
`tests/data/`.

## Settings

| Setting | Options | Default |
|---|---|---|
| Input genes | all genes in the gene table / a panel (`genes=`, `--hvg`) | all genes |
| Support of the predicted family measure | `genome` / `input` | `genome` |
| Precision | bfloat16 autocast / float32 (`amp=False`, `--fp32`) | bfloat16 |

Settings used in the paper:

* **Cross-species integration benchmarks.** 2,000 HVGs selected per species (scanpy `seurat` flavor on
  log-normalized counts, restricted to genes in the gene table), `support="input"`, bfloat16,
  batch size 128.
* **Predicted expression and virtual knockouts.** All genes, `support="genome"`, float32.

## Random seeds

Inference is deterministic. Pretraining used seed 42 for model initialization, sampling and masking.
The train/validation split used seed 12345 + 42.
