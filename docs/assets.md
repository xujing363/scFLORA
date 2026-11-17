# Pretrained model and assets

Weights and frozen assets are distributed separately from the code. Checksums are listed in
`docs/SHA256SUMS.txt`; verify the files with `sha256sum -c SHA256SUMS.txt`.

## `scflora_v1.pt` (36 MB)

A PyTorch file with the following keys:
* `state_dict`: 167 tensors, 9.03 M parameters;
* `config`: the `model`, `data`, `loss` and `train` sections; identical to `configs/scflora_v1_pretrain.json`;
* `step`: 200000.

The frozen ESM-2 tables are not part of the checkpoint. They are supplied per species by the gene tables.

## `family_space.npz` (154 MB)

* `centers`: [30000, 2560] float16 family centroids from spherical k-means on the L2-normalized
  ESM-2 embeddings of all 1,835,160 pretraining genes.
* `coarse_of_fine`: [30000] int64, the coarse stratum (one of 2,000) of each family. It is used only
  for stratified sampling.
* `k_coarse`: 2000.

## `genes/<species>.npz`

One file per pretraining genome annotation. Each file contains:
* `species`;
* `gene_ids`: canonical identifiers (see `scflora.canon_gene_id`);
* `esm`: [n, 2560] float16 ESM-2 3B embeddings, L2-normalized (the exact values used by the model);
* `family`: [n] int64 family of each gene.

For pretraining genes, `family` is the final spherical k-means assignment. For 95.9% of the genes it is
also the nearest centroid. Genes of a new species are assigned to the nearest centroid
(`FamilySpace.assign`). Rows are in the order of the pretraining vocabulary.

**ESM-2 recipe.**
* Longest protein isoform per gene.
* `facebook/esm2_t36_3B_UR50D` in float16, sequences truncated to 1,022 residues.
* Last hidden layer averaged over all tokens covered by the attention mask, including BOS and EOS.

This recipe is implemented in `scflora/protein.py`.

| Genome annotation (species key) | Genes | File size (MB) |
|---|---:|---:|
| Arabidopsis_lyrata | 31,066 | 161 |
| Bombax_ceiba | 26,394 | 138 |
| Brachypodium_distachyon | 39,053 | 203 |
| Brassica_oleracea | 35,112 | 182 |
| Brassica_rapa | 27,918 | 144 |
| Camelina_sativa | 66,045 | 342 |
| Camellia_sinensis | 50,525 | 261 |
| Capsella_rubella | 27,682 | 144 |
| Catharanthus_roseus | 20,749 | 107 |
| Chinesecabbage | 41,102 | 213 |
| Eucalyptus_grandis | 36,346 | 187 |
| Eutrema_salsugineum | 22,760 | 119 |
| Fragaria_vesca | 20,765 | 108 |
| Glycine_soja | 37,110 | 193 |
| Gossypium_bickii | 43,790 | 228 |
| Gossypium_hirsutum | 120,202 | 634 |
| Hevea_brasiliensis | 29,034 | 150 |
| Hygrophila_difformis | 24,127 | 125 |
| Lotus_japonicus | 25,243 | 132 |
| Manihot_esculenta | 22,507 | 117 |
| Manihot_esculenta_Crantz | 21,800 | 113 |
| Medicago_truncatula | 44,276 | 231 |
| Nepeta_tenuifolia | 20,354 | 105 |
| Nicotiana_attenuata | 25,094 | 130 |
| Petunia_x_hybrida | 22,573 | 118 |
| Phaseolus_vulgaris | 25,198 | 131 |
| Phragmites_australis | 36,473 | 191 |
| Phyllostachys_edulis | 50,547 | 262 |
| Pogostemon_cablin | 109,332 | 567 |
| Populus_alba | 28,465 | 147 |
| Populus_alba_var._pyramidalis | 28,909 | 150 |
| Populus_alba_x_Populus_glandulosa | 39,404 | 204 |
| Populus_alba_x_Populus_tremula | 32,255 | 167 |
| Populus_euramericana | 25,455 | 132 |
| Populus_tremula_x_Populus_alba | 26,306 | 137 |
| Populus_trichocarpa | 34,699 | 180 |
| Schrenkiella_parvula | 22,521 | 117 |
| Sisymbrium_irio | 23,233 | 121 |
| Solanum_lycopersicum | 52,219 | 271 |
| Sorghum_bicolor | 50,287 | 262 |
| Trochodendron_aralioides | 34,565 | 179 |
| Wolffia_australiana | 14,004 | 73 |
| arabidopsis | 27,628 | 143 |
| maize | 68,581 | 356 |
| rice | 65,131 | 337 |
| soybean | 52,941 | 275 |
| wheat | 105,380 | 548 |
| **total** | **1,835,160** | **9.5 GB** |
