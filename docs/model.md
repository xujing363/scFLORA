# Model and objective

This page maps the equations of the paper to the code. Symbols: cell *i* of species *s*, gene *g*,
protein-semantic family *f(g)*, raw counts *c<sub>ig</sub>*, depth *T<sub>i</sub>* = Σ<sub>g</sub> c<sub>ig</sub>.

## Gene and family coordinates (`model.py: ScFLORA.embed_vecs`)

* e<sub>g</sub>: ESM-2 3B embedding (mean over the last hidden layer). It is L2-normalized, stored as float16 and kept frozen.
* q<sub>g</sub> = LN(W<sub>2</sub> GELU(W<sub>1</sub> e<sub>g</sub> + b<sub>1</sub>) + b<sub>2</sub>), with d = 256.
* Family centroids c<sub>f</sub> go through the same map, so q<sub>f</sub> lies in the same space as the genes.
* Families come from spherical k-means at two resolutions (`families.py: spherical_kmeans`):
  * 30,000 fine families, which are the protein-semantic families;
  * 2,000 coarse clusters, used only as sampling strata.
* A gene of a new species is assigned to the fine family with the nearest centroid (`FamilySpace.assign`).

## Encoder: residual-measure attention (`model.py: ScFLORA.encode`, `RMABlock`)

**Observed measures.** Counts enter only as the normalized measure ĉ<sub>g</sub> = c<sub>g</sub>/T, so the
encoder is invariant to sequencing depth.
* The gene-level observed measure is ĉ<sub>g</sub>.
* The family-level observed measure is ĉ<sub>f</sub> = Σ<sub>g∈f</sub> ĉ<sub>g</sub>.

**Predicted measure.** In block *l* the current latent state P<sup>(l)</sup> is read out exactly as the
decoder reads the final state, h = W<sub>h</sub> u(LN P<sup>(l)</sup>). This gives a predicted family
measure over the families F<sub>s</sub> of the species' genome:

π<sub>l</sub>(f) ∝ exp(⟨A q<sub>f</sub>, h⟩/√d + b<sub>D</sub>(q<sub>f</sub>)).

* Parameters are shared with the decoder.
* The decoder's deformation is not applied here.
* At inference the sum runs over all families of F<sub>s</sub> (`support="genome"`). It can instead be
  limited to the families of the input genes (`support="input"`).
* During pretraining, 2,048 families are drawn per species by stratified sampling with log-weights
  log(n<sub>j</sub>/k<sub>j</sub>).

**Residual features.** With shared query/key/value projections, each latent token forms kernel means of
the coordinates under three measures:
* o<sub>fam</sub>: family coordinates under the observed family measure;
* p<sub>fam</sub>: family coordinates under the predicted family measure;
* o<sub>gene</sub>: gene coordinates under the observed gene measure.

The block update is

P ← P + W<sub>o</sub>[o<sub>fam</sub> − p<sub>fam</sub> ; o<sub>gene</sub> − o<sub>fam</sub> ; r], followed by self-attention and an FFN,

where r is the per-head difference of the observed and predicted log-partition functions.

**Configuration.** 64 latent tokens of width 256, 6 blocks, 8 heads.

**Cell embedding.** u = [P<sub>1</sub>; P<sub>2</sub>], 512-d (`cell_embedding`). It is the only input to the decoder.

## Decoder (`model.py: decoder_slots, deform, alloc_logit, density_shape, log_scale, log_rates`)

* **Slots.** u is expanded linearly into four key/value slots.
* **Deformation.** Two cross-attention blocks give context-dependent coordinates x̃ = x + Δ(x, u) for genes and families.
* **Readout vector.** h = W<sub>h</sub> u.
* **Family score.** shape(f) = ⟨A x̃<sub>f</sub>, h⟩/√d + b<sub>D</sub>(x̃<sub>f</sub>).
* **Within-family allocation score.** s(g) = ⟨A x̃<sub>g</sub>, h⟩/√d + b(x̃<sub>g</sub>).
* **Total rate.** log Λ(T) = log T + β + g(log T). It depends on depth only.
* **Gene rate.**
  log λ<sub>g</sub> = shape(f) − logsumexp<sub>f'∈F<sub>s</sub></sub> shape(f') + log Λ(T) + s(g) − logsumexp<sub>g'∈f, species s</sub> s(g').

So Σ<sub>g</sub> λ<sub>g</sub> = Λ(T) exactly. Polyploidy or larger gene families change how a family's rate is
split among its members, not the family's total.

## Objective (`loss.py: masked_nll`)

**Masking.** A random fraction ρ ~ U(0.3, 0.9) of each cell's positive-count genes is visible to the
encoder (`sample_visible`). The remaining positive-count genes H are targets.

**Negative log-likelihood**, per cell and divided by |H|:

−Σ<sub>g∈H</sub>[c<sub>g</sub> log λ<sub>g</sub> − log c<sub>g</sub>!] + Σ<sub>g∈H</sub> λ<sub>g</sub> + Σ<sub>g unobserved</sub> λ<sub>g</sub>

The last term covers genes with zero counts in the original cell.

**Computing the unobserved term.** It is computed per family as D(f)·(1 − Σ<sub>g∈f, observed</sub> a(g)):
* D(f) is the family rate and a(g) the within-family share;
* the residual factor is non-negative by construction.

**Sampling.** The term is estimated with 4,096 families per species drawn by stratified sampling. Strata
are the coarse clusters; k<sub>j</sub> is proportional to the stratum size and the weights are
n<sub>j</sub>/k<sub>j</sub>, which makes the estimator unbiased (`families.py: StratifiedSampler`). The same
sample estimates the family normalizer, so Σ<sub>f</sub> D(f) = Λ(T) holds in every batch.

## Readouts used for perturbations (`inference.py: virtual_knockout`)

* **Ranking score.** The mean over wild-type cells of |s<sub>KO</sub>(g) − s<sub>WT</sub>(g)|, i.e. the change
  in the decoder's within-family allocation logit after the target genes are removed from the input.
* **Signed response.** log<sub>2</sub>[Σ<sub>i</sub> λ<sub>ig</sub>(KO) / Σ<sub>i</sub> λ<sub>ig</sub>(WT)], with
  both arms evaluated at the wild-type depth.
