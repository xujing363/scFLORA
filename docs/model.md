# Model card

## Intended use

scFLORA generates cell representations and model-based expression responses from plant single-cell raw counts. It supports annotation, cross-species transfer, genome-wide rate prediction, and virtual perturbation prioritization.

## Architecture

Protein sequences are represented by frozen ESM-2 embeddings. Spherical k-means defines 30,000 fine protein-semantic families and 2,000 coarse strata. A six-block residual-measure attention encoder updates 64 latent tokens. The concatenation of the first two tokens is the 512-dimensional cell state. A family-factorized decoder predicts a non-negative rate for every annotated gene.

For a cell with depth `T`, the decoder enforces the exact decomposition

`sum_g lambda_g = Lambda(T)`

where family-level mass is normalized over the species support and within-family mass is normalized over member genes. The default `support="genome"` uses all families in the species gene table. `support="input"` is intended for predefined restricted panels.

## Readouts

- `embed`: 512-dimensional cell state.
- `predict_rates`: expected counts for every gene in the species table.
- `virtual_knockout`: allocation-score changes and signed gene-level log2 responses after zeroing target counts.

## Interpretation and limits

Outputs are model-based predictions, not causal measurements. Virtual knockout rankings prioritize candidates under the frozen model and input panel. They should be validated experimentally. Cross-species comparisons require matched preprocessing, split definitions, gene-table versions, and support settings. The checkpoint does not infer genes absent from the supplied species table.

## Numerical behavior

Use bfloat16 for throughput and fp32 when comparing small perturbation differences or producing audit figures. CPU and GPU outputs can differ by floating-point tolerance. The reference tests define the accepted numerical envelope.
