# Pretraining

## Inputs

1. **Gene tables** (`genes/<species>.npz`) for every pretraining species, together with
   `family_space.npz`. To build them for a new set of species:
   * embed each proteome with `scflora protein-embed`;
   * build the family space with `scflora.families.build_family_space` (spherical k-means:
     30,000 fine families and 2,000 coarse strata, 25 iterations, seed 42);
   * save each table with `GeneTable.save`.
2. **Species list** (`species.txt`): one species per line. Its order fixes the species ids.
3. **Corpus manifest** (TSV with columns `dataset_key`, `species` and `path`). Each `path` is an
   `.h5ad` whose `X` is a CSR matrix of **raw integer counts**.
4. Optional inputs:
   * `--col2tok-dir`: precomputed column-to-gene maps (`<dataset_key>.npy`; -1 means a column is not
     in the gene table). Without it, genes are matched by normalized identifier.
   * `--exclude-mask`: an `.npz` with a boolean `mask` over all corpus cells. It is used to remove
     cells that overlap evaluation datasets.

## Cell filters and split (`corpus.py`)

* A cell is kept if:
  * it has at least 200 genes with positive counts in the gene tables, and
  * its total count lies within ±3 SD of the mean of its dataset.
* 1% of the eligible cells of each dataset are held out for validation. The validation set is
  capped at 20,000 cells in total, split evenly across datasets.
* The split uses random seed `12345 + seed`.

## Sampling

* Each epoch draws as many cells as the training set, with replacement.
* Species s is drawn with probability ∝ n<sub>s</sub><sup>0.6</sup>; cells are drawn uniformly within a species.
* Draws are grouped into megabatches of 64 × 32 cells and sorted by length.
* Batches are cut under a token budget of 49,152 (cells × longest cell), with at most 128 cells per batch.
* Two micro-batches are accumulated per optimizer step.

## Optimization (`train.py`, `configs/scflora_v1_pretrain.json`)

* **Optimizer.** AdamW with β = (0.9, 0.95) and weight decay 0.1. Biases, normalization parameters and
  the latent tokens have no weight decay.
* **Learning rate.** Peak 3 × 10⁻⁴, linear warm-up over 2% of the steps, cosine decay to 10⁻⁵.
* **Numerics and regularization.** bfloat16 autocast, TF32 matmuls, gradient clipping at 1.0, dropout 0.1.
* **Masking and partition term.** Visible fraction ~ U(0.3, 0.9); 4,096 sampled families for the
  partition term; 2,048 families for the encoder's predicted measure.
* **Schedule.** The schedule covers 15 epochs. The released checkpoint was taken at step 200,000,
  within the second epoch.

## Commands

```bash
scflora pack  --manifest corpus.tsv --gene-dir assets/genes --species species.txt \
              --family-space assets/family_space.npz --cache-dir run/cache --pack-dir pack
scflora train --config configs/scflora_v1_pretrain.json --manifest corpus.tsv \
              --gene-dir assets/genes --species species.txt --family-space assets/family_space.npz \
              --pack-dir pack --out-dir run --devices 0          # e.g. --devices 0,1,2 for DDP
```

* A checkpoint and a resumable state (`resume.pt`) are written every `save_every` steps. Re-running the
  same command resumes from `resume.pt`.
* `--max-steps N` stops early, which is useful for smoke tests.
* If DataLoader workers fail with `AF_UNIX path too long`, set `TMPDIR` to a short path.

**Resources for the released model.** One H100 80 GB GPU, with peak memory of about 34 GB. The packed
corpus takes about 105 GB.

**Smoke test.** On 2 species × 4,000 cells, the package trains at about 2.5 optimizer steps per second
on one H100.
