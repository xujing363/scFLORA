"""scFLORA model: protein-informed gene representations, a residual-measure
attention (RMA) encoder and a family-factorized generative decoder.

Notation (per cell i of species s, genes g, families f):
  e_g          frozen ESM-2 embedding of gene g (L2-normalized)
  x_g          = LN(MLP(e_g)), gene coordinate; x_f = LN(MLP(c_f)) for family centroid c_f
  c_ig, T_i    observed count and depth; c_hat = c / T (normalized measure)
  P            latent tokens [M, d]; cell embedding u = concat(P_1, P_2) (512-d)

Decoder (rates of every protein-coding gene of the species):
  h            = W_h u
  x~           = x + Delta(x, u): two cross-attention blocks whose keys/values are
                 n_dec_slots vectors linearly expanded from u
  s(g)         = <A x~_g, h>/sqrt(d) + b(x~_g)                      (within-family score)
  shape(f)     = <A x~_f, h>/sqrt(d) + b_D(x~_f)                    (family score)
  log D(f)     = shape(f) - logsumexp_{f' in F_s} shape(f') + log Lambda(T)
  log Lambda   = log T + beta + g(log T)                            (depends on depth only)
  log lambda_g = log D(f(g)) + s(g) - logsumexp_{g' in f(g), species s} s(g')

Encoder (six RMA blocks): latent queries attend to (i) the observed family
measure, (ii) the family measure predicted from the current latent state with
the decoder's own family score, and (iii) the observed gene measure; the
differences between these summaries update the latent tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .families import FamilySpace, GenomeIndex, GenomeMeasure


@dataclass
class PointBatch:
    """A batch of cells as variable-length sets of (gene token, count), right-padded."""
    tok: Tensor        # [B, N] long, token ids in the current GenomeIndex
    cnt: Tensor        # [B, N] float counts
    valid: Tensor      # [B, N] bool
    species: Tensor    # [B] long, species position in the current GenomeIndex
    depth: Tensor      # [B] float, total in-vocabulary counts

    def to(self, dev, non_blocking: bool = False) -> "PointBatch":
        return PointBatch(**{f.name: getattr(self, f.name).to(dev, non_blocking=non_blocking)
                             for f in fields(self)})


class MHA(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh, self.p = n_heads, d // n_heads, dropout
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def _split(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dh).transpose(1, 2)

    def forward(self, xq: Tensor, xkv: Tensor, key_valid: Optional[Tensor] = None) -> Tensor:
        q, k, v = self._split(self.q(xq)), self._split(self.k(xkv)), self._split(self.v(xkv))
        mask = None if key_valid is None else key_valid[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.p if self.training else 0.0)
        B, h, L, dh = o.shape
        return self.o(o.transpose(1, 2).reshape(B, L, h * dh))


class FFN(nn.Module):
    def __init__(self, d: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """Gene/family coordinates attend to the cell-embedding slots."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        d, h, p = c.d_model, c.n_heads, c.dropout
        self.ln_q, self.ln_z = nn.LayerNorm(d), nn.LayerNorm(d)
        self.cross = MHA(d, h, p)
        self.ln_q2 = nn.LayerNorm(d)
        self.ffn = FFN(d, c.d_ff, p)

    def forward(self, q: Tensor, z: Tensor) -> Tensor:
        q = q + self.cross(self.ln_q(q), self.ln_z(z))
        return q + self.ffn(self.ln_q2(q))


class _KernelMean(torch.autograd.Function):
    """Chunked weighted attention mean and log-partition.

    q [B,h,M,dh]; k, v [S,h,L,dh]; inv [B] (row of k/v for each cell) or None (S == B);
    logw [B,L] (-inf = excluded).
      out[b,:,m] = sum_j softmax_j(q.k_j/sqrt(dh) + logw_j) v_j
      lse[b,:,m] = logsumexp_j(q.k_j/sqrt(dh) + logw_j)
    Computed in float32 (float64 inputs stay float64) with autocast disabled;
    the backward pass recomputes the softmax chunk by chunk.
    """

    @staticmethod
    def forward(ctx, q, k, v, inv, logw, chunk):
        ctx.set_materialize_grads(False)
        wd = torch.float64 if q.dtype == torch.float64 else torch.float32
        ctx.dts = (q.dtype, k.dtype, v.dtype, logw.dtype)
        with torch.autocast(device_type=q.device.type, enabled=False):
            q_, k_, v_, lw = q.to(wd), k.to(wd), v.to(wd), logw.to(wd)
            B, H, M, dh = q_.shape
            L = k_.shape[2]
            scale = 1.0 / math.sqrt(dh)
            step = max(1, int(chunk) // max(1, H * M * L))
            out = q_.new_empty(B, H, M, v_.shape[-1])
            lse = q_.new_empty(B, H, M)
            for a in range(0, B, step):
                b = min(a + step, B)
                kc = k_[a:b] if inv is None else k_[inv[a:b]]
                vc = v_[a:b] if inv is None else v_[inv[a:b]]
                lg = torch.matmul(q_[a:b], kc.transpose(-1, -2)) * scale + lw[a:b, None, None, :]
                ls = torch.logsumexp(lg, dim=-1)
                A = torch.exp(lg - ls.unsqueeze(-1))
                del lg
                out[a:b] = torch.matmul(A, vc)
                lse[a:b] = ls
        ctx.save_for_backward(q_, k_, v_, lw, lse, inv if inv is not None else torch.empty(0))
        ctx.has_inv = inv is not None
        ctx.step, ctx.scale = step, scale
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, lw, lse, inv = ctx.saved_tensors
        need_lw = ctx.needs_input_grad[4]
        with torch.autocast(device_type=q.device.type, enabled=False):
            B = q.shape[0]
            dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
            dlw = torch.zeros_like(lw) if need_lw else None
            if dout is not None:
                dout = dout.to(q.dtype)
            if dlse is not None:
                dlse = dlse.to(q.dtype)
            for a in range(0, B, ctx.step):
                b = min(a + ctx.step, B)
                if ctx.has_inv:
                    ix = inv[a:b]
                    kc, vc = k[ix], v[ix]
                else:
                    kc, vc = k[a:b], v[a:b]
                lg = torch.matmul(q[a:b], kc.transpose(-1, -2)) * ctx.scale + lw[a:b, None, None, :]
                A = torch.exp(lg - lse[a:b].unsqueeze(-1))
                del lg
                dL = None
                if dout is not None:
                    do = dout[a:b]
                    dvc = torch.matmul(A.transpose(-1, -2), do)
                    if ctx.has_inv:
                        dv.index_add_(0, ix, dvc)
                    else:
                        dv[a:b] = dvc
                    dA = torch.matmul(do, vc.transpose(-1, -2))
                    dL = A * (dA - (A * dA).sum(-1, keepdim=True))
                if dlse is not None:
                    t = A * dlse[a:b].unsqueeze(-1)
                    dL = t if dL is None else dL + t
                if dL is None:
                    continue
                dq[a:b] = torch.matmul(dL, kc) * ctx.scale
                dkc = torch.matmul(dL.transpose(-1, -2), q[a:b]) * ctx.scale
                if ctx.has_inv:
                    dk.index_add_(0, ix, dkc)
                else:
                    dk[a:b] = dkc
                if need_lw:
                    dlw[a:b] = dL.sum(dim=(1, 2))
        d0, d1, d2, d3 = ctx.dts
        return dq.to(d0), dk.to(d1), dv.to(d2), None, (None if dlw is None else dlw.to(d3)), None


class RMABlock(nn.Module):
    """Residual-measure attention block.

    o_fam  = kernel mean of family coordinates under the observed family measure
    p_fam  = kernel mean of family coordinates under the predicted family measure
    o_gene = kernel mean of gene coordinates under the observed gene measure
    r      = log-partition(observed families) - log-partition(predicted families)
    P <- P + W_o [o_fam - p_fam ; o_gene - o_fam ; r]  -> self-attention -> FFN
    Query, key and value projections are shared by the three branches.
    """

    def __init__(self, c: ModelConfig):
        super().__init__()
        d, h, p = c.d_model, c.n_heads, c.dropout
        self.h, self.dh, self.chunk = h, d // h, int(c.rma_chunk)
        self.ln_p1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(2 * d + h, d)
        self.ln_p2 = nn.LayerNorm(d)
        self.self_attn = MHA(d, h, p)
        self.ln_p3 = nn.LayerNorm(d)
        self.ffn = FFN(d, c.d_ff, p)

    def _split(self, x: Tensor) -> Tensor:
        *lead, L, _ = x.shape
        return x.reshape(*lead, L, self.h, self.dh).transpose(-3, -2)

    def residual_features(self, P: Tensor, xf: Tensor, lwf: Tensor, xg: Tensor, lwg: Tensor,
                          xs: Tensor, inv: Tensor, lwp: Tensor) -> Tensor:
        B, M, _ = P.shape
        q = self._split(self.q(self.ln_p1(P)))
        o_f, lse_f = _KernelMean.apply(q, self._split(self.k(xf)), self._split(self.v(xf)),
                                       None, lwf, self.chunk)
        p_f, lse_p = _KernelMean.apply(q, self._split(self.k(xs)), self._split(self.v(xs)),
                                       inv, lwp, self.chunk)
        o_g, _ = _KernelMean.apply(q, self._split(self.k(xg)), self._split(self.v(xg)),
                                   None, lwg, self.chunk)

        def merge(t: Tensor) -> Tensor:
            return t.transpose(1, 2).reshape(B, M, -1)

        return torch.cat([merge(o_f - p_f), merge(o_g - o_f), (lse_f - lse_p).transpose(1, 2)], dim=-1)

    def forward(self, P, xf, lwf, xg, lwg, xs, inv, lwp) -> Tensor:
        P = P + self.o(self.residual_features(P, xf, lwf, xg, lwg, xs, inv, lwp))
        P = P + self.self_attn(self.ln_p2(P), self.ln_p2(P))
        return P + self.ffn(self.ln_p3(P))


class ScFLORA(nn.Module):
    """scFLORA. Gene identity enters only through frozen protein embeddings, so the
    same weights apply to any species once its ``GenomeIndex`` is attached
    (``set_genome``)."""

    def __init__(self, cfg: ModelConfig, space: FamilySpace):
        super().__init__()
        self.c = cfg
        d = cfg.d_model
        self.k_fine = space.k_fine
        # protein-informed gene representation
        self.proj_e = nn.Sequential(nn.Linear(cfg.esm_dim, d), nn.GELU(), nn.Linear(d, d))
        self.ln_e = nn.LayerNorm(d)
        # encoder
        self.latents = nn.Parameter(torch.randn(cfg.n_latents, d) * 0.02)
        self.enc = nn.ModuleList([RMABlock(cfg) for _ in range(cfg.n_enc_layers)])
        self.ln_z = nn.LayerNorm(d)
        # decoder
        self.decoder = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_dec_layers)])
        self.ln_dec = nn.LayerNorm(d)
        self.head_A = nn.Linear(d, d, bias=False)
        self.head_h = nn.Linear(cfg.embedding_dim, d, bias=False)
        self.head_gene_bias = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1, bias=False))
        self.log_beta = nn.Parameter(torch.zeros(1))
        self.head_family_bias = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1, bias=False))
        self.to_dec = nn.Linear(cfg.embedding_dim, cfg.n_dec_slots * d, bias=False)
        self.head_depth = nn.Sequential(nn.Linear(1, max(d // 4, 8)), nn.GELU(), nn.Linear(max(d // 4, 8), 1))
        nn.init.normal_(self.head_depth[-1].weight, std=1e-3)
        nn.init.zeros_(self.head_depth[-1].bias)
        # family centroids (fixed); gene tables are attached with set_genome()
        Cf = F.normalize(torch.as_tensor(np.asarray(space.centers, dtype=np.float32)), dim=1)
        self.register_buffer("centers", Cf.to(torch.float16), persistent=False)
        self.__dict__["_measure"] = None
        self.__dict__["_family_space"] = space
        self.genome: Optional[GenomeIndex] = None

    # ── genome / species ──────────────────────────────────────────────────────
    def set_genome(self, index: GenomeIndex, device=None, measure_samples: Optional[int] = None) -> None:
        """Attach the gene tables of one or more species (frozen; not saved)."""
        dev = device if device is not None else self.log_beta.device
        E = torch.as_tensor(np.asarray(index.esm, dtype=np.float16))      # unit-norm rows
        self.register_buffer("esm", E.to(dev), persistent=False)
        for name in ("fine_of_tok", "fam_cnt", "fam_ptr", "fam_mem", "rank_in_fam"):
            self.register_buffer(name, torch.from_numpy(getattr(index, name)).long().to(dev),
                                 persistent=False)
        self.__dict__["_measure"] = GenomeMeasure(
            index, measure_samples or self.c.rma_samples, dev)
        self.genome = index

    def set_measure_support(self, families: Optional[np.ndarray] = None, species: int = 0) -> None:
        """Support of the encoder's predicted family measure at inference.

        None (default): all families of the species' genome, as in pretraining.
        An array: only these families (e.g. the families of the genes in an
        input gene panel, so that observed and predicted measures share support).
        """
        measure = self.__dict__.get("_measure")
        if measure is None:
            raise RuntimeError("call model.set_genome(...) first")
        if families is None:
            fresh = GenomeMeasure(self.genome, self.c.rma_samples, self.log_beta.device)
            measure.full_fid, measure.full_mask = fresh.full_fid, fresh.full_mask
        else:
            measure.restrict(species, families)

    # ── coordinates ───────────────────────────────────────────────────────────
    def embed_vecs(self, e: Tensor) -> Tensor:
        return self.ln_e(self.proj_e(e.to(self.proj_e[0].weight.dtype)))

    def embed_tokens(self, tok: Tensor) -> Tensor:
        return self.embed_vecs(F.embedding(tok, self.esm))

    def embed_family(self, fid: Tensor) -> Tensor:
        return self.embed_vecs(self.centers[fid])

    # ── encoder ───────────────────────────────────────────────────────────────
    def encode(self, tok: Tensor, cnt: Tensor, valid: Tensor, species: Tensor,
               generator: Optional[torch.Generator] = None) -> Tensor:
        """Point sets -> latent tokens [B, M, d]. Counts enter only as the normalized
        measure c/T, so the representation is invariant to sequencing depth."""
        measure = self.__dict__.get("_measure")
        if measure is None:
            raise RuntimeError("call model.set_genome(...) first")
        B, N = tok.shape
        kf, dev, NEG = self.k_fine, tok.device, float("-inf")
        wdt = torch.float64 if self.proj_e[0].weight.dtype == torch.float64 else torch.float32
        c = (cnt * valid).to(wdt)
        pos = valid & (c > 0)
        if not bool(pos.any(1).all()):
            raise RuntimeError("every cell needs at least one positive-count gene")
        chat = c / c.sum(1, keepdim=True).clamp(min=1e-12)
        # observed family measure: one entry per family (first position), mass = family sum
        fam = self.fine_of_tok[tok].masked_fill(~pos, kf)
        fs, order = torch.sort(fam, dim=1, stable=True)
        cs = torch.cumsum(chat.gather(1, order).double(), dim=1)
        ar = torch.arange(N, device=dev).expand(B, N)
        first = torch.ones_like(fs, dtype=torch.bool)
        first[:, 1:] = fs[:, 1:] != fs[:, :-1]
        last = torch.ones_like(fs, dtype=torch.bool)
        last[:, :-1] = fs[:, 1:] != fs[:, :-1]
        start = torch.cummax(torch.where(first, ar, torch.zeros_like(ar)), dim=1).values
        end = torch.flip(torch.cummin(torch.flip(torch.where(last, ar, torch.full_like(ar, N - 1)), [1]),
                                      dim=1).values, [1])
        seg = cs.gather(1, end) - torch.where(start > 0, cs.gather(1, (start - 1).clamp(min=0)),
                                              torch.zeros_like(cs))
        keep = first & (fs < kf)
        lwf_s = torch.where(keep, torch.log(seg.clamp(min=1e-300)).to(wdt), torch.full_like(chat, NEG))
        lwf = torch.full_like(chat, NEG).scatter(1, order, lwf_s)
        xf = self.embed_family(fam.clamp(max=kf - 1))
        # observed gene measure
        xg = self.embed_tokens(tok)
        lwg = torch.where(pos, torch.log(chat.clamp(min=1e-30)), torch.full_like(chat, NEG))
        # support of the predicted family measure: the species' families
        uq, inv = torch.unique(species, return_inverse=True)
        fid, lw_s = measure.get(uq, training=self.training, generator=generator)
        xs = self.embed_family(fid) if self.training else self.embed_family(torch.arange(kf, device=dev))[fid]
        As = self.head_A(xs)
        bDs = self.head_family_bias(xs).squeeze(-1)
        sd = math.sqrt(self.c.d_model)
        P = self.latents.unsqueeze(0).expand(B, -1, -1)
        for blk in self.enc:
            h = self.head_h(self.cell_embedding(self.ln_z(P)))       # decoder readout of the current state
            lwp = lw_s[inv].to(wdt) + ((As[inv] * h.unsqueeze(1)).sum(-1) / sd + bDs[inv]).to(wdt)
            lwp = lwp - torch.logsumexp(lwp, dim=1, keepdim=True)    # log predicted family measure
            P = blk(P, xf, lwf, xg, lwg, xs, inv, lwp)
        return self.ln_z(P)

    def cell_embedding(self, z: Tensor) -> Tensor:
        """512-d cell embedding: the first ``n_summary`` latent tokens, concatenated.
        It is the only input of the decoder."""
        return z[:, :self.c.n_summary].reshape(z.shape[0], -1)

    def forward(self, batch: PointBatch, visible: Optional[Tensor] = None,
                generator: Optional[torch.Generator] = None) -> Dict[str, Tensor]:
        """Encode the visible genes (all genes when ``visible`` is None)."""
        valid = batch.valid if visible is None else (batch.valid & visible)
        cnt_v = batch.cnt * valid
        z = self.encode(batch.tok, cnt_v, valid, batch.species, generator)
        return {"z": z, "cell_emb": self.cell_embedding(z), "enc_valid": valid,
                "depth_eff": cnt_v.sum(dim=1)}

    # ── decoder primitives ────────────────────────────────────────────────────
    def decoder_slots(self, z: Tensor) -> Tensor:
        return self.to_dec(self.cell_embedding(z)).view(z.shape[0], self.c.n_dec_slots, self.c.d_model)

    def deform(self, q0: Tensor, slots: Tensor) -> Tensor:
        """Context-conditioned coordinates x~ = x + Delta(x, u)."""
        q = q0
        for blk in self.decoder:
            q = blk(q, slots)
        return self.ln_dec(q)

    def warp(self, q0: Tensor, z: Tensor) -> Tensor:
        return self.deform(q0, self.decoder_slots(z))

    def readout(self, z: Tensor) -> Tensor:
        return self.head_h(self.cell_embedding(z))

    def alloc_logit(self, e_t: Tensor, h: Tensor) -> Tensor:
        """Within-family score s(g | u) (depth-free; normalized within each family)."""
        return ((self.head_A(e_t) * h.unsqueeze(1)).sum(-1) / math.sqrt(self.c.d_model)
                + self.head_gene_bias(e_t).squeeze(-1))

    def density_shape(self, e_f: Tensor, h: Tensor) -> Tensor:
        """Family score shape(f | u) (normalized over the species' families)."""
        return ((self.head_A(e_f) * h.unsqueeze(1)).sum(-1) / math.sqrt(self.c.d_model)
                + self.head_family_bias(e_f).squeeze(-1))

    def log_scale(self, depth: Tensor, log_norm: Tensor) -> Tensor:
        """log Lambda(T) - log Z_F, shape [B, 1]: total rate depends only on depth."""
        logT = torch.log(depth.clamp(min=1.0)).unsqueeze(1)
        return logT + self.log_beta + self.head_depth(logT).squeeze(-1).unsqueeze(1) - log_norm.unsqueeze(1)

    def family_members_chunked(self, tok: Tensor, valid: Tensor, species: Tensor, C: int):
        """All same-species members of the families of each cell's observed genes,
        laid out as blocks of width C (each block belongs to one cell).

        Returns memb/mfam/mmask [Q, C], cellof [Q], slot [B, N] (index of each observed
        gene in the flattened [Q*C] layout) and fam [B, N].
        """
        B, N = tok.shape
        dev, kf = tok.device, self.k_fine
        fam = self.fine_of_tok[tok].masked_fill(~valid, kf)
        fam_s, _ = fam.sort(dim=1)
        first = torch.ones_like(fam_s, dtype=torch.bool)
        first[:, 1:] = fam_s[:, 1:] != fam_s[:, :-1]
        keep = first & (fam_s < kf)
        famc = fam_s.clamp(max=kf - 1)
        cnt = torch.where(keep, self.fam_cnt[species.unsqueeze(1), famc], torch.zeros_like(fam_s))
        tot = cnt.sum(1)
        tot_c = tot.to("cpu", non_blocking=False)
        nch_c = torch.div(tot_c + (C - 1), C, rounding_mode="floor")
        Ftot = int(tot_c.sum().item())
        Q = max(1, int(nch_c.sum().item()))
        choff = (torch.cumsum(nch_c, 0) - nch_c).to(dev)
        cflat = cnt.reshape(-1)
        ar = torch.arange(Ftot, device=dev)
        src = torch.repeat_interleave(torch.arange(B * N, device=dev), cflat, output_size=Ftot)
        rank = ar - (torch.cumsum(cflat, 0) - cflat)[src]
        cell = torch.div(src, N, rounding_mode="floor")
        fsrc = famc.reshape(-1)[src]
        mtok = self.fam_mem[self.fam_ptr[species[cell], fsrc] + rank]
        cell_off = torch.cumsum(tot, 0) - tot
        posn = choff[cell] * C + (ar - cell_off[cell])
        memb = tok.new_zeros(Q * C)
        mfam = tok.new_full((Q * C,), kf)
        mmask = torch.zeros(Q * C, dtype=torch.bool, device=dev)
        memb[posn] = mtok
        mfam[posn] = fsrc
        mmask[posn] = True
        cellof = (torch.searchsorted(choff, torch.arange(Q, device=dev), right=True) - 1).clamp(0, B - 1)
        b0 = rank == 0
        blk = tok.new_zeros(B, kf + 1)
        blk[cell[b0], fsrc[b0]] = posn[b0]
        slot = (blk.gather(1, fam) + self.rank_in_fam[tok]).masked_fill(~valid, 0)
        return (memb.view(Q, C), mfam.view(Q, C), mmask.view(Q, C), cellof,
                slot.clamp(min=0, max=Q * C - 1), fam)

    # ── genome-wide predicted rates (exact) ──────────────────────────────────
    @torch.no_grad()
    def log_rates(self, z: Tensor, depth: Tensor, species: int = 0, chunk: int = 1024
                  ) -> Tuple[Tensor, Tensor]:
        """Exact log predicted rates of every gene of ``species`` for cells ``z``.

        The family normalizer is summed over all families of the genome (not
        sampled). Returns (log_lambda [B, G], token ids [G]) in family-member order.
        """
        dev = z.device
        cnt = self.fam_cnt[species].cpu().numpy()
        ptr = self.fam_ptr[species].cpu().numpy()
        mem = self.fam_mem.cpu().numpy()
        fams = np.flatnonzero(cnt > 0)
        gtok = np.concatenate([mem[ptr[f]:ptr[f] + cnt[f]] for f in fams])
        gfam = np.repeat(np.arange(len(fams)), cnt[fams])
        G, Fn = len(gtok), len(fams)
        gtok_t = torch.as_tensor(gtok, device=dev)
        gfam_t = torch.as_tensor(gfam, device=dev)
        fams_t = torch.as_tensor(fams, device=dev)
        dt = torch.float64 if self.log_beta.dtype == torch.float64 else torch.float32
        z = z.to(dt)
        B = z.shape[0]
        h = self.readout(z).to(dt)
        slots = self.decoder_slots(z)
        s = torch.empty(B, G, device=dev, dtype=dt)
        for a in range(0, G, chunk):
            q = self.embed_tokens(gtok_t[a:a + chunk].unsqueeze(0).expand(B, -1)).to(dt)
            s[:, a:a + chunk] = self.alloc_logit(self.deform(q, slots), h).to(dt)
        idx = gfam_t.unsqueeze(0).expand(B, -1)
        mx = torch.full((B, Fn), -1e30, device=dev, dtype=dt).scatter_reduce_(1, idx, s, reduce="amax", include_self=True)
        ex = torch.exp(s - mx.gather(1, idx))
        lse = mx + torch.log(torch.zeros(B, Fn, device=dev, dtype=dt).scatter_add_(1, idx, ex).clamp_min(1e-30))
        log_a = s - lse.gather(1, idx)
        sh = torch.empty(B, Fn, device=dev, dtype=dt)
        for a in range(0, Fn, chunk):
            qf = self.embed_family(fams_t[a:a + chunk]).unsqueeze(0).expand(B, -1, -1).to(dt)
            sh[:, a:a + chunk] = self.density_shape(self.deform(qf, slots), h).to(dt)
        scale = self.log_scale(depth.to(dt), torch.logsumexp(sh, 1)).to(dt).reshape(B, 1)
        ll = (sh.gather(1, idx) + log_a + scale).clamp(-self.c.log_rate_clamp, self.c.log_rate_clamp)
        return ll, gtok_t
