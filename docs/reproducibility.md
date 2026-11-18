# Reproducibility protocol

This repository reproduces **inference from the released checkpoint**. It does not claim to reproduce pretraining. Every published result should be accompanied by a machine-readable run manifest containing the fields below.

## Required manifest

```yaml
repository_commit: <git commit>
checkpoint_sha256: <sha256>
family_space_sha256: <sha256>
gene_table_sha256: <sha256>
input_sha256: <sha256>
command: <complete command line>
software: environment.yml
python: "3.10"
torch: "2.6.0"
device: <GPU model or CPU>
precision: bf16|fp32
seed: 42
n_cells: <integer>
n_input_genes: <integer>
```

## Input contract

- Matrix orientation is cells × genes.
- Values are non-negative raw counts. Do not pass log-normalized values.
- `var_names` must match the canonical identifiers in the species table after version and prefix normalization.
- Preserve cell order, gene order, and observation identifiers in the output.
- For comparisons, freeze the split, QC thresholds, HVG panel, target list, and metric before running scFLORA or any comparator.

## Analysis settings

| Analysis | Input support | Precision | Output validation |
|---|---|---|---|
| Full-genome embedding | `genome` | bf16 or fp32 | shape `(n_cells, 512)` |
| HVG integration | `input` | bf16 | identical panel definition across methods |
| Predicted expression | genome | fp32 recommended | rates non-negative and depth-consistent |
| Virtual knockout | genome | fp32 recommended | same wild-type cells and depth in both arms |

The encoder is deterministic for a fixed device, precision, input order, and checkpoint. GPU and CPU floating-point results can differ within the tolerances tested in `tests/test_pretrained.py`.

## Verification gates

1. Verify all SHA-256 checksums.
2. Run `pytest tests -q`.
3. Run `python examples/demo.py --assets assets/`.
4. Confirm the demo reports `matches reference: True`.
5. Archive the manifest and output checksum with the result.

## Reporting perturbation analyses

Report the wild-type cell count, target genes, target detection rate, candidate universe, whether cells were encoded jointly, depth handling, and the exact score definition. Do not compare rankings produced with different cell panels, species gene tables, or support settings.

## Known scope boundaries

The checkpoint is frozen. The public package does not expose pretraining, corpus construction, or manuscript-specific analysis orchestration. Those materials are not needed for downstream inference and are not represented as reproducible by this release.
