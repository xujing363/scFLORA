import math

import numpy as np
import torch

from scflora.config import DataConfig, LossConfig
from scflora.families import StratifiedSampler
from scflora.loss import masked_nll, sample_visible
from scflora.model import _KernelMean

from conftest import toy_batch


def test_kernel_mean_matches_naive_and_gradcheck():
    torch.manual_seed(0)
    B, H, M, L, dh = 3, 2, 4, 7, 5
    q = torch.randn(B, H, M, dh, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, H, L, dh, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, H, L, dh, dtype=torch.float64, requires_grad=True)
    inv = torch.tensor([0, 1, 0])
    logw = torch.randn(B, L, dtype=torch.float64)
    logw[0, -2:] = float("-inf")
    logw.requires_grad_(True)
    out, lse = _KernelMean.apply(q, k, v, inv, logw, 10)
    lg = torch.matmul(q, k[inv].transpose(-1, -2)) / math.sqrt(dh) + logw[:, None, None, :]
    assert torch.allclose(lse, torch.logsumexp(lg, -1))
    assert torch.allclose(out, torch.matmul(torch.softmax(lg, -1), v[inv]))
    f = lambda q_, k_, v_, lw_: _KernelMean.apply(q_, k_, v_, inv, lw_, 10)
    lw_finite = logw.detach().clone(); lw_finite[0, -2:] = -5.0
    lw_finite.requires_grad_(True)
    assert torch.autograd.gradcheck(f, (q, k, v, lw_finite))


def test_stratified_sampler_is_unbiased():
    rng = np.random.default_rng(3)
    stratum = rng.integers(0, 8, size=200)
    bank = np.sort(rng.choice(200, 120, replace=False))
    s = StratifiedSampler([bank], stratum, 8, n_samples=64)
    x = torch.from_numpy(rng.random(200) + 0.05)
    exact = float(x[torch.from_numpy(bank)].sum())
    g = torch.Generator().manual_seed(11)
    vals = []
    for _ in range(2000):
        fid, w = s.sample(torch.zeros(1, dtype=torch.long), generator=g)
        vals.append(float((x[fid] * w).sum()))
    m, se = np.mean(vals), np.std(vals, ddof=1) / math.sqrt(len(vals))
    assert abs(m - exact) <= 3 * se


def test_family_index_tables(toy):
    model, index = toy
    for s in range(len(index.species)):
        toks = np.arange(index.offsets[s], index.offsets[s + 1])
        for t in toks[:10]:
            f = index.fine_of_tok[t]
            blk = index.fam_mem[index.fam_ptr[s, f]: index.fam_ptr[s, f] + index.fam_cnt[s, f]]
            assert blk[index.rank_in_fam[t]] == t
        assert np.array_equal(index.banks[s], np.unique(index.fine_of_tok[toks]))


def test_embedding_invariances(toy):
    model, index = toy
    b = toy_batch(index)
    with torch.no_grad():
        e = model(b)["cell_emb"]
        b2 = toy_batch(index); b2.cnt = b2.cnt * 7.0; b2.depth = b2.cnt.sum(1)
        assert torch.allclose(e, model(b2)["cell_emb"], atol=1e-10)          # depth invariance
        perm = torch.randperm(8)
        b3 = toy_batch(index); b3.tok[0, :8] = b3.tok[0, :8][perm]; b3.cnt[0, :8] = b3.cnt[0, :8][perm]
        assert torch.allclose(e, model(b3)["cell_emb"], atol=1e-10)          # gene-order invariance
        b4 = toy_batch(index, lengths=(8,), species=(0,))
        assert torch.allclose(e[:1], model(b4)["cell_emb"], atol=1e-10)      # padding invariance


def test_total_rate_depends_on_depth_only(toy):
    model, index = toy
    b = toy_batch(index, species=(0, 0, 0))
    with torch.no_grad():
        o = model(b)
        ll, gtok = model.log_rates(o["z"], o["depth_eff"], species=0)
        ll = ll.double()
        logT = torch.log(o["depth_eff"]).unsqueeze(1)
        lam_total = torch.exp(logT + model.log_beta + model.head_depth(logT)).squeeze(1)
    assert len(gtok) == index.offsets[1] - index.offsets[0]
    assert torch.allclose(torch.exp(ll).sum(1), lam_total, rtol=1e-8)


def test_masked_likelihood_gradient(toy):
    """Analytic vs numerical directional derivative of the loss (float64)."""
    model, index = toy
    b = toy_batch(index)
    vis = b.valid.clone(); vis[:, ::2] = False; vis &= b.valid
    sampler = index.family_sampler(16).double()
    lc, dc = LossConfig(family_samples=16), DataConfig()

    def f():
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            return float(masked_nll(model, b, sampler, lc, dc, generator=g, visible=vis)[0])

    model.zero_grad()
    g = torch.Generator().manual_seed(7)
    masked_nll(model, b, sampler, lc, dc, generator=g, visible=vis)[0].backward()
    params = [p for p in model.parameters() if p.requires_grad]
    rng = torch.Generator().manual_seed(1)
    vs = [torch.randn(p.shape, generator=rng, dtype=p.dtype) for p in params]
    nrm = math.sqrt(sum(float((v ** 2).sum()) for v in vs))
    vs = [v / nrm for v in vs]
    ana = sum(float((p.grad * v).sum()) for p, v in zip(params, vs))
    base = [p.data.clone() for p in params]

    def D(h):
        for p, b0, v in zip(params, base, vs):
            p.data.copy_(b0 + h * v)
        fp = f()
        for p, b0, v in zip(params, base, vs):
            p.data.copy_(b0 - h * v)
        fm = f()
        for p, b0 in zip(params, base):
            p.data.copy_(b0)
        return (fp - fm) / (2 * h)

    num = (4 * D(5e-5) - D(1e-4)) / 3
    assert abs(ana - num) / max(abs(ana), abs(num)) < 1e-7
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in params)


def test_visible_mask_keeps_a_target():
    valid = torch.ones(5, 9, dtype=torch.bool); valid[0, 2:] = False
    v = sample_visible(valid, 0.3, 0.9, torch.Generator().manual_seed(0))
    assert (v.sum(1) >= 1).all() and ((valid & ~v).sum(1) >= 1).all()
