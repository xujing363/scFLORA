"""Masked conditional Poisson point-process likelihood over the genome.

For each cell a random subset V of its positive-count genes is shown to the
encoder (visible fraction rho ~ U(0.3, 0.9)); the remaining positive-count
genes H are targets. The likelihood is evaluated on the whole genome except V:

  l = sum_{g in H} [c_g log lambda_g - log c_g!] - sum_{g in H} lambda_g - sum_{g unobserved} lambda_g

The last term (genes with zero counts in the original cell) is computed per
family: sum over unobserved members of family f = D(f) (1 - sum_{g in f, observed} a(g)),
estimated with a stratified sample of the species' families. The same sample
estimates the family normalizer, so sum_f D(f) = Lambda(T) holds exactly in
every batch. The loss is the negative log-likelihood divided by |H| and
averaged over cells.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .config import DataConfig, LossConfig
from .families import StratifiedSampler
from .model import PointBatch, ScFLORA


def sample_visible(valid: Tensor, lo: float, hi: float,
                   generator: Optional[torch.Generator] = None) -> Tensor:
    """Per cell, keep round(rho * n) random genes visible (rho ~ U(lo, hi)), with
    at least one visible and one hidden gene."""
    B, N = valid.shape
    dev = valid.device
    u = torch.rand(B, N, device=dev, generator=generator).masked_fill(~valid, 2.0)
    order = u.argsort(dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(N, device=dev).expand(B, N))
    n = valid.sum(1, keepdim=True)
    rho = lo + (hi - lo) * torch.rand(B, 1, device=dev, generator=generator)
    k = (rho * n.float()).round().long().clamp(min=1)
    k = torch.minimum(k, (n - 1).clamp(min=1))
    return valid & (rank < k)


def masked_nll(model: ScFLORA, batch: PointBatch, sampler: StratifiedSampler,
               lcfg: LossConfig, dcfg: DataConfig,
               generator: Optional[torch.Generator] = None,
               visible: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, float]]:
    if visible is None:
        visible = sample_visible(batch.valid, dcfg.mask_ratio_lo, dcfg.mask_ratio_hi, generator)
    out = model(batch, visible=visible, generator=generator)
    z, depth_eff = out["z"], out["depth_eff"]
    hidden = batch.valid & (~visible)
    B, N = batch.tok.shape
    dev, kf, clamp = batch.tok.device, model.k_fine, model.c.log_rate_clamp
    acc = torch.float64 if model.log_beta.dtype == torch.float64 else torch.float32   # never below fp32
    h = model.readout(z)
    slots = model.decoder_slots(z)

    # (1) within-family allocation a(g) for every observed gene
    C = int(model.c.member_chunk)
    memb, mfam, mmask, cellof, slot, fam = model.family_members_chunked(
        batch.tok, batch.valid, batch.species, C)
    s_m = model.alloc_logit(model.deform(model.embed_tokens(memb), slots[cellof]), h[cellof]).to(acc)
    NEG = -1e9
    s_masked = torch.where(mmask, s_m, torch.full_like(s_m, NEG))
    idx = (cellof.unsqueeze(1) * (kf + 1) + mfam).reshape(-1)
    mx = torch.full((B * (kf + 1),), NEG, device=dev, dtype=s_m.dtype)
    mx.scatter_reduce_(0, idx, s_masked.detach().reshape(-1), reduce="amax", include_self=True)
    ex = torch.exp(s_masked - mx[idx].view_as(s_masked)) * mmask
    Zf = torch.zeros(B * (kf + 1), device=dev, dtype=ex.dtype).scatter_add_(0, idx, ex.reshape(-1))
    lse = (mx + torch.log(Zf.clamp_min(1e-30))).view(B, kf + 1)
    log_a = s_m.reshape(-1)[slot.reshape(-1)].view(B, N) - lse.gather(1, fam)
    log_a = log_a.masked_fill(~batch.valid, -60.0)      # padding never reaches exp()

    # (2) family normalizer and family rates, from one stratified family sample
    fam_c = fam.clamp(max=kf - 1)
    e_fp = model.deform(model.embed_family(fam_c), slots)
    uq, iv = torch.unique(batch.species, return_inverse=True)
    fu, wu = sampler.sample(uq, generator=generator)
    samp_fid, samp_w = fu[iv], wu[iv]
    samp_shape = model.density_shape(model.deform(model.embed_family(samp_fid), slots), h).to(acc)
    log_norm = torch.logsumexp(samp_shape + torch.log(samp_w.clamp(min=1e-30)), dim=1)
    log_D_pt = (model.density_shape(e_fp, h) + model.log_scale(depth_eff, log_norm)).to(acc)
    log_lam = (log_D_pt + log_a).clamp(-clamp, clamp)

    # (3) hidden observed genes: exact
    hf, cf = hidden.to(acc), batch.cnt.to(acc)
    pos = ((cf * log_lam - torch.lgamma(cf + 1.0)) * hf).sum(1)
    lam_hidden = (log_lam.exp() * hf).sum(1)

    # (4) unobserved genes: sampled families x non-negative within-family residual
    a_pt = (log_a.exp() * batch.valid.to(acc)).to(acc)
    A_obs = torch.zeros(B, kf + 1, device=dev, dtype=a_pt.dtype).scatter_add_(1, fam, a_pt)
    log_D_s = (samp_shape + model.log_scale(depth_eff, log_norm)).to(acc).clamp(max=clamp)
    resid = (1.0 - A_obs.gather(1, samp_fid)).clamp(min=0.0)
    lam_unobs = (log_D_s.exp() * resid * samp_w).sum(1)

    n_hidden = hf.sum(1).clamp(min=1.0)
    loss = ((-pos + lcfg.w_partition * (lam_hidden + lam_unobs)) / n_hidden).mean()
    with torch.no_grad():
        stats = {"nll": float(loss),
                 "pos": float((-pos / n_hidden).mean()),
                 "lam_hidden": float((lam_hidden / n_hidden).mean()),
                 "lam_unobs": float((lam_unobs / n_hidden).mean()),
                 "n_hidden": float(n_hidden.mean()),
                 "rate_total_over_T": float(((lam_hidden + lam_unobs) / depth_eff.clamp(min=1)).mean())}
    return loss, stats


class LossModule(torch.nn.Module):
    """Model + sampler + likelihood in one module, so that DistributedDataParallel
    sees the full graph."""

    def __init__(self, model: ScFLORA, sampler: StratifiedSampler, lcfg: LossConfig, dcfg: DataConfig):
        super().__init__()
        self.model, self.sampler, self.lcfg, self.dcfg = model, sampler, lcfg, dcfg

    def forward(self, batch: PointBatch, generator: Optional[torch.Generator] = None):
        return masked_nll(self.model, batch, self.sampler, self.lcfg, self.dcfg, generator=generator)
