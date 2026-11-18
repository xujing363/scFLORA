# Assets and data contract

The code repository contains the checkpoint. Large frozen assets are distributed through the versioned release archive and must be checked before use.

## Files

| File | Role | Required by |
|---|---|---|
| `scflora_v1.pt` | 9.03M-parameter checkpoint, step 200,000 | all inference |
| `family_space.npz` | 30,000 protein-semantic family centroids and 2,000 strata | all inference |
| `genes/<species>.npz` | canonical gene IDs, normalized ESM-2 vectors, family IDs | species inference |

Run from the repository root:

```bash
sha256sum -c scflora_v1.pt.sha256
(cd assets && sha256sum -c ../docs/SHA256SUMS.txt)
```

Never mix a gene table or family space from another checkpoint release.

## Gene-table schema

Each species file contains `species`, `gene_ids`, `esm`, and `family`. `esm` is an L2-normalized float16 matrix with one row per gene. `family` is the frozen family assignment used by the decoder. The row order is part of the artifact and must not be sorted after download.

## New species

`scflora protein-embed` creates a compatible table from a protein FASTA. Use one longest isoform per gene, record the FASTA checksum, and retain the ESM-2 model identifier and revision. New genes are assigned to the nearest released family centroid. This transfer path does not alter the checkpoint.

## Storage and provenance

Keep the original downloaded files immutable. Store checksums, release version, download URL, and date in the run manifest. Do not commit private patient data or generated result matrices to this repository.
