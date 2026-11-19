"""
Two control objectives, same channels: move the BEHAVIOUR, or move the whole STATE.

The SURF report says the inferred channels had "negligible authority over a
128-D state", which suggests its LQR target was a full latent state. That is
a far harder objective than steering one behavioural readout: k inputs over a
short window cannot place an N-dimensional state at an arbitrary target, no
matter where they point. If so, the 0.1% partly reflects the objective chosen,
not the steerability of behaviour.

STATE OBJECTIVE
---------------
Over the delay window, the end-of-window state responds to inputs through
M_t = ∂h(T_end)/∂x(t) (computed exactly by autograd). Through channels B the
state controllability Gramian is

    W_B = Σ_t (M_t B)(M_t B)ᵀ

and for a target displacement e the LQR-style controller removes the fraction
eᵀ W_B (W_B + ρI)⁻¹ e / eᵀe. Averaged over random target directions:

    ER_state(B) = tr(W_B (W_B + ρI)⁻¹) / N

ρ is set per trial so that actuating every unit (B = I) removes 90% on
average — the same normalisation used for the behavioural objective, where ρ
is set so full actuation removes 90% of the behavioural error.
"""

import json
import numpy as np
import torch

import control as C
import steer as S


def solve_rho(evals, target=0.9):
    """ρ with mean(λ/(λ+ρ)) = target, by bisection in log space."""
    evals = np.clip(evals, 0, None)
    lo, hi = 1e-12, evals.max() * 1e6 + 1e-9
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if np.mean(evals / (evals + mid)) > target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def er_state(Ms, B, rho):
    """Ms: (Tw, N, N) = M_t. Average state error reduction through channels B."""
    MB = np.einsum("tij,jk->tik", Ms, B)
    W = np.einsum("tik,tjk->ij", MB, MB)
    ev = np.linalg.eigvalsh((W + W.T) / 2)
    return float(np.mean(np.clip(ev, 0, None) / (np.clip(ev, 0, None) + rho)))


def er_behaviour(Q, B, rho):
    g = float(np.trace(B.T @ Q @ B))
    return g / (g + rho)


def end_state_jacobians(step_fn, n, T, N, idx, t_end):
    """
    step_fn(x) runs the network with additive pre-activation input x (n, T, N)
    and returns the state trajectory (n, T, N). Returns, per trial, the stack
    M_t = ∂h(t_end)/∂x(t) over its window indices, shape (n, Tw, N, N).
    """
    x = torch.zeros(n, T, N, requires_grad=True)
    H = step_fn(x)
    hend = torch.stack([H[i, t_end[i]] for i in range(n)])     # (n, N)
    Tw = len(idx[0])
    out = np.zeros((n, Tw, N, N), np.float32)
    for unit in range(N):
        gx, = torch.autograd.grad(hend[:, unit].sum(), x, retain_graph=True)
        gx = gx.numpy()
        for i in range(n):
            out[i, :, unit, :] = gx[i, idx[i]]
    return out


def analyse(Ms, Qs, channel_sets, n_random=50, seed=0):
    rng = np.random.default_rng(seed)
    N = Ms.shape[-1]
    res = {name: {"state": [], "behaviour": []} for name in channel_sets}
    res["random"] = {"state": [], "behaviour": []}
    res["opt_behaviour"] = {"state": [], "behaviour": []}
    for i in range(len(Ms)):
        W_I = np.einsum("tij,tkj->ik", Ms[i], Ms[i])
        rho_s = solve_rho(np.linalg.eigvalsh((W_I + W_I.T) / 2))
        rho_b = np.trace(Qs[i]) / 9.0
        k = None
        for name, B in channel_sets.items():
            P = C.orth(B); k = P.shape[1]
            res[name]["state"].append(er_state(Ms[i], P, rho_s))
            res[name]["behaviour"].append(er_behaviour(Qs[i], P, rho_b))
        rs, rb = [], []
        for _ in range(n_random):
            P = np.linalg.qr(rng.standard_normal((N, k)))[0]
            rs.append(er_state(Ms[i], P, rho_s)); rb.append(er_behaviour(Qs[i], P, rho_b))
        res["random"]["state"].append(np.mean(rs)); res["random"]["behaviour"].append(np.mean(rb))
        ev, V = np.linalg.eigh((Qs[i] + Qs[i].T) / 2)
        P = V[:, -k:]
        res["opt_behaviour"]["state"].append(er_state(Ms[i], P, rho_s))
        res["opt_behaviour"]["behaviour"].append(er_behaviour(Qs[i], P, rho_b))
    return {name: {obj: dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                             q75=float(np.percentile(v, 75))) for obj, v in d.items()}
            for name, d in res.items()}, k


# ------------------------------------------------------------------ drivers
def run_student(path, data, n_trials=64, ridge=1e-2):
    from student_steer import load_student, lever_decoder
    st, ck = load_student(path, data)
    w, b, _ = lever_decoder(st, data, ridge)
    tg = data["t_go"]; tl = data["t_load"]
    s, _ = S.student_sensitivity(st, w, b, tg)
    sel = np.arange(n_trials)
    idx = [list(range(t - 30, t - 5)) for t in tg[sel]]
    t_end = np.array([t - 5 for t in tg[sel]])
    g0 = torch.tanh(st.g0.detach())[sel]; U = st.U.detach()[sel]
    Bm = st.B().detach(); Wm = st.W.detach(); bm = st.b.detach()

    def step_fn(x):
        g = g0; G = []
        for t in range(st.T):
            g = torch.tanh(g @ Wm.T + bm + U[:, t] @ Bm.T + x[:, t]); G.append(g)
        return torch.stack(G, 1)
    Ms = end_state_jacobians(step_fn, len(sel), st.T, st.N, idx, t_end)
    Qs = S.Q_from(s[sel], idx)
    res, k = analyse(Ms, Qs, {"fit": Bm.numpy()})
    return dict(path=path, k=k, lam=ck["lam_u"], N=st.N, result=res)


def run_teacher(path, n_trials=64, seed=21):
    from teacher import load_teacher, cued_response
    net, _ = load_teacher(path)
    rng = np.random.default_rng(seed)
    u, z, m, meta = cued_response(n_trials, rng=rng, return_meta=True)
    tg = meta["t_go"]
    s, _ = S.teacher_sensitivity(net, u, tg)
    idx = [list(range(t - 30, t - 5)) for t in tg]
    t_end = np.array([t - 5 for t in tg])

    def step_fn(x):
        h = torch.zeros(n_trials, net.n_hid); H = []
        for t in range(u.shape[1]):
            h = torch.tanh(h @ net.W_rec.T + u[:, t] @ net.W_in.T + net.b + x[:, t]); H.append(h)
        return torch.stack(H, 1)
    Ms = end_state_jacobians(step_fn, n_trials, u.shape[1], net.n_hid, idx, t_end)
    Qs = S.Q_from(s, idx)
    Win = net.W_in.detach().numpy()
    res, k = analyse(Ms, Qs, {"real_inputs": Win})
    return dict(path=path, k=k, N=net.n_hid, result=res)


if __name__ == "__main__":
    import sys
    torch.set_num_threads(1)
    out = {"teacher": run_teacher("runs/teacher_g0.9_n0.05_s0.pt")}
    data = torch.load("runs/dataset.pt", weights_only=False)
    for p in sys.argv[1:]:
        out[p] = run_student(p, data)
    json.dump(out, open("runs/state_target.json", "w"), indent=1)
    for key, r in out.items():
        print(f"\n{key}  (k={r['k']}, N={r['N']})")
        for name, d in r["result"].items():
            print(f"   {name:15s} behaviour {d['behaviour']['median']*100:5.1f}%   whole state {d['state']['median']*100:5.1f}%")
