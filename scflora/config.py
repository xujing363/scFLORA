"""Configuration objects for scFLORA.

The defaults reproduce the released pretrained model (scFLORA v1): 64 latent
tokens of width 256, six residual-measure attention (RMA) blocks with eight
heads, two decoder cross-attention blocks and a 512-dimensional cell embedding
(the first two final latent tokens).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Tuple


@dataclass
class ModelConfig:
    esm_dim: int = 2560            # ESM-2 (3B) protein embedding width
    d_model: int = 256             # latent token width
    n_heads: int = 8
    d_ff: int = 832                # feed-forward width
    n_latents: int = 64            # number of trainable latent tokens
    n_summary: int = 2             # cell embedding = first n_summary latent tokens
    n_enc_layers: int = 6          # RMA blocks
    n_dec_layers: int = 2          # decoder cross-attention blocks
    n_dec_slots: int = 4           # decoder key/value slots expanded from the cell embedding
    dropout: float = 0.1
    log_rate_clamp: float = 20.0   # |log lambda| clamp for numerical stability
    rma_samples: int = 2048        # families sampled per species for the predicted measure (training only)
    rma_chunk: int = 67108864      # max logits per chunk in the RMA kernel (memory only)
    member_chunk: int = 128        # block width of the family-member layout in the loss (memory only)

    @property
    def embedding_dim(self) -> int:
        return self.n_summary * self.d_model


@dataclass
class DataConfig:
    mask_ratio_lo: float = 0.3     # visible fraction of positive-count genes ~ U(lo, hi)
    mask_ratio_hi: float = 0.9
    min_expressed_genes: int = 200 # pretraining cell filter (genes with positive counts in the vocabulary)
    outlier_sd: float = 3.0        # pretraining cell filter on total counts, per dataset
    val_frac: float = 0.01
    val_cap: int = 20000
    bucket_mult: int = 64          # length bucketing: megabatch = bucket_mult x batch_size
    max_tokens: int = 49152        # per-batch budget of (cells x max expressed genes)
    max_batch: int = 128
    n_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True


@dataclass
class LossConfig:
    family_samples: int = 4096     # families sampled per species for the partition term
    w_partition: float = 1.0


@dataclass
class TrainConfig:
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    batch_size: int = 32           # nominal batch size used for length bucketing
    accumulate: int = 2
    max_epochs: int = 2
    precision: str = "bf16"        # bf16 | fp32
    species_alpha: float = 0.6     # species sampling p_s proportional to n_s ** alpha
    seed: int = 42
    log_every: int = 50
    save_every: int = 20000
    limit_val_batches: int = 50


def to_dict(*cfgs) -> Dict[str, Any]:
    names = {ModelConfig: "model", DataConfig: "data", LossConfig: "loss", TrainConfig: "train"}
    return {names[type(c)]: asdict(c) for c in cfgs}


def from_dict(cls, d: Dict[str, Any]):
    """Build a config, ignoring unknown keys and refusing missing required ones."""
    known = {f.name for f in fields(cls)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in known}
    return cls(**kw)
