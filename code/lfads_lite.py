"""
A minimal LFADS — the STUDENT, fit to spikes from the teacher.

It reproduces the parts of LFADS that the SURF steering result depends on:

    g(t+1) = tanh(W g(t) + B u(t) + b)        generator, N = 64 units
    f(t)   = F g(t)                            12 factors
    rate   = exp(Dout f(t) + d)                Poisson observations
    g(0), u(t)  inferred per trial             initial condition + controller

and nothing else. Departures from real LFADS, deliberately:

  * Per-trial g(0) and u(t) are FREE PARAMETERS optimised directly, not
    produced by an encoder network. The quantity of interest is the fitted
    generator (W, B), not amortised inference, and this removes a layer that
    could otherwise be blamed for whatever we find.
  * The controller penalty is an L2 cost on u, lam_u * mean(u²) — the Gaussian
    analogue of LFADS's KL term on the controller output. lam_u plays the
    role of kl_co_scale: small = cfg01-like, large = cfg06-like.
  * Columns of B are held at unit norm. Without this the model can inflate B
    and shrink u, sidestepping the penalty entirely, and lam_u would mean
    nothing.
  * Vanilla tanh generator, not a GRU, so the Jacobian is exact and matches
    everything else in this project.

The student never sees the teacher's inputs (the value pulse or the go cue),
its outputs, or its states. Only spikes.
"""

import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn

from teacher import cued_response, run_noisy, load_teacher, T_TRIAL


# ---------------------------------------------------------------------------
# data: Poisson spikes from the teacher
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_dataset(teacher, n_trials=256, n_obs=60, n_heldout=10, seed=7,
                 mod_sd=0.7, rate_range=(0.08, 0.45)):
    """
    Each recorded neuron's log-rate is a random projection of the teacher
    state, rescaled so its modulation has s.d. mod_sd and its mean rate sits in
    rate_range (spikes per 20 ms bin, i.e. 4-22 Hz). Mixed selectivity, like
    real cortex.
    """
    rng = np.random.default_rng(seed)
    u, z, m, meta = cued_response(n_trials, rng=rng, return_meta=True)
    _, H = run_noisy(teacher, u, 0.0)
    H = H.numpy()                                              # (n, T, 100)
    C = rng.standard_normal((n_obs, H.shape[-1]))
    X = H.reshape(-1, H.shape[-1]) @ C.T                       # (n*T, n_obs)
    X = (X - X.mean(0)) / X.std(0)
    base = rng.uniform(*rate_range, n_obs)
    logr = np.log(base) + mod_sd * X - mod_sd ** 2 / 2
    rates = np.exp(logr).reshape(n_trials, -1, n_obs)
    spikes = rng.poisson(rates).astype(np.float32)
    return dict(spikes=torch.from_numpy(spikes), rates=rates.astype(np.float32),
                H_teacher=H.astype(np.float32), u_teacher=u.numpy(), z=z.numpy(),
                v=meta["v"], t_go=meta["t_go"], t_load=meta["t_load"],
                n_heldin=n_obs - n_heldout, C=C)


# ---------------------------------------------------------------------------
# the student
# ---------------------------------------------------------------------------

class Student(nn.Module):
    def __init__(self, n_trials, T, n_obs, N=64, n_fac=12, k=4, g=0.9, seed=0,
                 mean_rate=None):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.N, self.k, self.T = N, k, T
        self.W = nn.Parameter(torch.randn(N, N, generator=gen) * g / math.sqrt(N))
        self.Braw = nn.Parameter(torch.randn(N, k, generator=gen))
        self.b = nn.Parameter(torch.zeros(N))
        self.F = nn.Parameter(torch.randn(n_fac, N, generator=gen) / math.sqrt(N))
        self.Dout = nn.Parameter(torch.randn(n_obs, n_fac, generator=gen) * 0.3)
        d0 = torch.log(mean_rate) if mean_rate is not None else torch.zeros(n_obs)
        self.d = nn.Parameter(d0.clone())
        self.g0 = nn.Parameter(torch.randn(n_trials, N, generator=gen) * 0.1)
        self.U = nn.Parameter(torch.zeros(n_trials, T, k))

    def B(self):
        return self.Braw / self.Braw.norm(dim=0, keepdim=True)

    def generate(self, use_u=True, g0=None, U=None):
        g = torch.tanh(self.g0) if g0 is None else g0
        U = self.U if U is None else U
        B = self.B()
        G, P = [], []
        for t in range(self.T):
            pre = g @ self.W.T + self.b
            if use_u:
                pre = pre + U[:, t] @ B.T
            g = torch.tanh(pre)
            G.append(g); P.append(pre)
        return torch.stack(G, 1), torch.stack(P, 1)

    def logrates(self, G):
        return (G @ self.F.T) @ self.Dout.T + self.d


def poisson_nll(logr, s):
    return (torch.exp(logr) - s * logr).mean()


def bits_per_spike(logr, s):
    """Information gain over a per-neuron constant-rate model, in bits/spike."""
    mu = s.mean(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    nll_model = (torch.exp(logr) - s * logr).sum()
    nll_null = (mu - s * torch.log(mu)).sum()
    return ((nll_null - nll_model) / (s.sum() * math.log(2))).item()


def fit_student(data, k=4, lam_u=1e-2, N=64, steps=2500, lr=1e-2, seed=0,
                lam_w=1e-5, log_every=250):
    torch.manual_seed(seed)
    S = data["spikes"]
    nin = data["n_heldin"]
    Sin = S[..., :nin]
    n, T, _ = S.shape
    st = Student(n, T, nin, N=N, k=k, seed=seed,
                 mean_rate=Sin.mean(dim=(0, 1)).clamp(min=1e-3))
    opt = torch.optim.Adam(st.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    t0 = time.time()
    for it in range(1, steps + 1):
        G, _ = st.generate()
        lr_ = st.logrates(G)
        nll = poisson_nll(lr_, Sin)
        loss = nll + lam_u * (st.U ** 2).mean() + lam_w * (st.W ** 2).sum() \
            + 1e-3 * (st.g0 ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(st.parameters(), 5.0)
        opt.step(); sched.step()
        if it % log_every == 0 or it == 1:
            print(f"  step {it:5d}  nll {nll.item():.4f}  |u|^2 {(st.U ** 2).mean().item():.4f}"
                  f"  bps {bits_per_spike(lr_.detach(), Sin):.4f}  ({time.time() - t0:.0f}s)",
                  flush=True)
    return st


# ---------------------------------------------------------------------------
# evaluation — the SURF metrics, now with ground truth available
# ---------------------------------------------------------------------------

def fit_glm(X, Y, steps=400, l2=1e-3):
    """Poisson GLM, X (n,T,p) -> Y (n,T,q). Returns predicted log-rates."""
    Xf = X.reshape(-1, X.shape[-1]); Yf = Y.reshape(-1, Y.shape[-1])
    Wt = torch.zeros(X.shape[-1], Y.shape[-1], requires_grad=True)
    bt = torch.log(Yf.mean(0).clamp(min=1e-3)).clone().requires_grad_(True)
    o = torch.optim.LBFGS([Wt, bt], max_iter=steps, line_search_fn="strong_wolfe")
    def closure():
        o.zero_grad()
        lr = Xf @ Wt + bt
        l = (torch.exp(lr) - Yf * lr).mean() + l2 * (Wt ** 2).sum()
        l.backward(); return l
    o.step(closure)
    with torch.no_grad():
        return (Xf @ Wt + bt).reshape(Y.shape)


@torch.no_grad()
def evaluate(st, data):
    S = data["spikes"]; nin = data["n_heldin"]
    G, P = st.generate(use_u=True)
    Ga, _ = st.generate(use_u=False)
    lr_in = st.logrates(G)
    fac = G @ st.F.T

    out = {}
    out["bps_heldin"] = bits_per_spike(lr_in, S[..., :nin])
    # co-smoothing: held-out neurons predicted from inferred factors
    with torch.enable_grad():
        lr_out = fit_glm(fac, S[..., nin:])
    out["bps_cosmooth"] = bits_per_spike(lr_out, S[..., nin:])

    # autonomy: generator run from the same g(0) with u = 0, vs the full model
    def r2(a, b):
        a = a.reshape(-1, a.shape[-1]); b = b.reshape(-1, b.shape[-1])
        return 1 - ((a - b) ** 2).sum().item() / ((a - a.mean(0)) ** 2).sum().item()
    out["autonomy"] = r2(G, Ga)
    t_go = data["t_go"]
    pre = torch.zeros(G.shape[:2], dtype=torch.bool)
    for i, tg in enumerate(t_go):
        pre[i, :tg] = True
    Gp, Gap = G[pre], Ga[pre]
    Gq, Gaq = G[~pre], Ga[~pre]
    out["autonomy_pre_go"] = 1 - ((Gp - Gap) ** 2).sum().item() / ((Gp - Gp.mean(0)) ** 2).sum().item()
    out["autonomy_post_go"] = 1 - ((Gq - Gaq) ** 2).sum().item() / ((Gq - Gq.mean(0)) ** 2).sum().item()
    out["u_power"] = (st.U ** 2).mean().item()

    # does the inferred controller recover the hidden cue? (SURF surprise 1)
    U = st.U.detach().norm(dim=-1).numpy()                      # (n, T)
    ua = np.stack([U[i, tg - 20:tg + 40] for i, tg in enumerate(t_go)])
    out["u_go_locked"] = ua.mean(0).tolist()                     # -20..+39 around go
    ul = np.stack([U[i, tl - 5:tl + 25] for i, tl in enumerate(data["t_load"])])
    out["u_load_locked"] = ul.mean(0).tolist()
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--lam_u", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="runs/dataset.pt")
    p.add_argument("--suffix", default="")
    a = p.parse_args()
    torch.set_num_threads(1)
    data = torch.load(a.data, weights_only=False)
    tag = f"student_k{a.k}_lam{a.lam_u}_s{a.seed}{a.suffix}"
    print(f"{tag}", flush=True)
    st = fit_student(data, k=a.k, lam_u=a.lam_u, steps=a.steps, seed=a.seed)
    ev = evaluate(st, data)
    print(json.dumps({k: v for k, v in ev.items() if not isinstance(v, list)}, indent=2))
    torch.save({"state_dict": st.state_dict(), "k": a.k, "lam_u": a.lam_u,
                "seed": a.seed, "N": st.N, "eval": ev}, f"runs/{tag}.pt")
    with open(f"runs/{tag}_eval.json", "w") as f:
        json.dump(ev, f)
