"""
Causal steering analysis — what can input through a channel set do to BEHAVIOUR?

WHY THIS REPLACES THE DECODER VERSION
-------------------------------------
The first version defined "the behavioural axis" as a ridge decoder of the
plan from delay-period states. On the teacher that failed its positive
control, and the reason is general: delay states live on a nearly 1-D line,
so a decoder can put large weight on near-empty directions and still decode
perfectly. Its share in the slow subspace moved from 1% to 77% with the ridge
parameter while decoding R² stayed at 0.999. A decoder does not define the
direction that matters for behaviour.

So behaviour is defined CAUSALLY here: run the network forward with an
additive input x(t) entering the pre-activation, and differentiate the actual
behaviour with respect to it,

    s(t) = ∂ amp / ∂ x(t),       amp = mean lever output over the plateau
                                       of the pull (go+8 .. go+28)

through the full nonlinear trajectory of that trial. Input through channels B
enters as x(t) = B u(t), so ∂amp/∂u(t) = Bᵀ s(t) and

    G(B) = Σ_{t in window} ‖Bᵀ s(t)‖² = tr(Bᵀ Q B),     Q = Σ s(t) s(t)ᵀ

The same Gramian, the same exact LQR identity (error reduction = G/(G+ρ)),
but no decoder and no fixed-point linearisation.

WINDOWS
-------
  load:  the 5 steps of the value pulse — POSITIVE CONTROL. The value channel
         is how the plan gets set, so here it must score well.
  delay: 25 steps ending 5 steps before the go cue — THE QUESTION. Can input
         during the hold change what the animal will do?

For the student the behaviour is the student's own predicted lever (a ridge
decoder from its generator state to the lever, fit during the pull where the
lever actually varies), differentiated back through the student's generator
with its inferred u held at the fitted values. Robustness to that decoder's
ridge is checked explicitly, given what happened above.
"""

import math
import numpy as np
import torch

import control as C


# ---------------------------------------------------------------------------
# sensitivity Gramians
# ---------------------------------------------------------------------------

def plateau_mask(t_go, T):
    m = torch.zeros(len(t_go), T)
    for i, tg in enumerate(t_go):
        m[i, tg + 8:tg + 28] = 1.0 / 20
    return m


def window_idx(kind, t_load, t_go):
    if kind == "load":
        return [list(range(tl, tl + 5)) for tl in t_load]
    if kind == "delay":
        return [list(range(tg - 30, tg - 5)) for tg in t_go]
    raise ValueError(kind)


def teacher_sensitivity(net, u, t_go):
    """Returns s (B, T, N): ∂amp/∂x(t) for an additive pre-activation input."""
    B, T, _ = u.shape
    N = net.n_hid
    x = torch.zeros(B, T, N, requires_grad=True)
    h = torch.zeros(B, N)
    ys = []
    for t in range(T):
        h = torch.tanh(h @ net.W_rec.T + u[:, t] @ net.W_in.T + net.b + x[:, t])
        ys.append(h @ net.W_out[0] + net.b_out[0])
    y = torch.stack(ys, 1)
    amp = (y * plateau_mask(t_go, T)).sum(1)
    amp.sum().backward()
    return x.grad.detach().numpy(), amp.detach().numpy()


def student_sensitivity(st, w_lev, b_lev, t_go):
    """Same, through the student's generator with its fitted g0 and U."""
    n, T, k = st.U.shape
    N = st.N
    x = torch.zeros(n, T, N, requires_grad=True)
    g = torch.tanh(st.g0.detach())
    U = st.U.detach(); Bm = st.B().detach(); W = st.W.detach(); b = st.b.detach()
    ys = []
    for t in range(T):
        g = torch.tanh(g @ W.T + b + U[:, t] @ Bm.T + x[:, t])
        ys.append(g @ w_lev + b_lev)
    y = torch.stack(ys, 1)
    amp = (y * plateau_mask(t_go, T)).sum(1)
    amp.sum().backward()
    return x.grad.detach().numpy(), amp.detach().numpy()


def Q_from(s, idx):
    """Per-trial Q matrices over the given window indices."""
    return [s[i, w].T @ s[i, w] for i, w in enumerate(idx)]


# ---------------------------------------------------------------------------
# comparing channel sets
# ---------------------------------------------------------------------------

def compare(Qs, channel_sets, k_list=(1, 2, 3, 4, 6, 8, 12, 16), n_random=200, seed=0):
    """
    Per trial: G for each named channel set (as an orthonormal span), the
    random distribution at matching k, the optimum at matching k, and G_full.
    Returns medians over trials plus the pooled optimal-k curve.
    """
    rng = np.random.default_rng(seed)
    per = {name: [] for name in channel_sets}
    curves = []
    for Q in Qs:
        Gfull = float(np.trace(Q))
        if Gfull <= 0:
            continue
        ev = np.linalg.eigvalsh((Q + Q.T) / 2)[::-1]
        curves.append([ev[:k].sum() / Gfull for k in k_list])
        rho = Gfull / 9.0                      # full actuation would remove 90%
        for name, Bm in channel_sets.items():
            P = C.orth(Bm); k = P.shape[1]
            g = C.G_of(Q, P)
            gr = C.G_random(Q, k, n_random, rng)
            go = float(ev[:k].sum())
            per[name].append(dict(
                aim=(g - gr.mean()) / (go - gr.mean()),
                pct=float((gr < g).mean() * 100),
                ratio_rand=g / gr.mean(),
                frac_full=g / Gfull,
                frac_full_opt=go / Gfull,
                err_red=C.error_reduction(g, rho),
                err_red_rand=C.error_reduction(gr.mean(), rho),
                err_red_opt=C.error_reduction(go, rho),
                Gfull=Gfull))
    out = {}
    for name, rows in per.items():
        out[name] = {key: dict(median=float(np.median([r[key] for r in rows])),
                               q25=float(np.percentile([r[key] for r in rows], 25)),
                               q75=float(np.percentile([r[key] for r in rows], 75)))
                     for key in rows[0]}
    out["opt_curve"] = dict(k=list(k_list), median=np.median(curves, 0).tolist(),
                            q25=np.percentile(curves, 25, 0).tolist(),
                            q75=np.percentile(curves, 75, 0).tolist())
    return out


def pooled_optimal_direction(Qs):
    Qp = sum(Qs) / len(Qs)
    ev, V = np.linalg.eigh(Qp)
    return V[:, -1], ev[::-1]


# ---------------------------------------------------------------------------
# linearity check: does the Gramian predict what the nonlinear network does?
# ---------------------------------------------------------------------------

@torch.no_grad()
def teacher_amp_with_input(net, u, t_go, x):
    B, T, _ = u.shape
    h = torch.zeros(B, net.n_hid); ys = []
    for t in range(T):
        h = torch.tanh(h @ net.W_rec.T + u[:, t] @ net.W_in.T + net.b + x[:, t])
        ys.append(h @ net.W_out[0] + net.b_out[0])
    y = torch.stack(ys, 1)
    return (y * plateau_mask(t_go, T)).sum(1).numpy()


def linearity_check(net, u, t_go, s, idx, eps_list=(0.01, 0.03, 0.1, 0.3, 1.0)):
    """
    Apply the minimum-energy input that the linear analysis says should change
    amp by eps, for the full-actuation case (x(t) ∝ s(t) on the window), run the
    real network, and compare achieved to predicted change.
    """
    B, T, N = s.shape
    base = teacher_amp_with_input(net, u, t_go, torch.zeros(B, T, N))
    rows = []
    for eps in eps_list:
        x = np.zeros((B, T, N), np.float32)
        for i, w in enumerate(idx):
            G = float((s[i, w] ** 2).sum())
            x[i, w] = eps * s[i, w] / G           # min-energy input for Δamp = eps
        amp = teacher_amp_with_input(net, u, t_go, torch.from_numpy(x))
        achieved = amp - base
        rows.append(dict(eps=eps, achieved_median=float(np.median(achieved)),
                         ratio_median=float(np.median(achieved / eps)),
                         ratio_q25=float(np.percentile(achieved / eps, 25)),
                         ratio_q75=float(np.percentile(achieved / eps, 75))))
    return rows
