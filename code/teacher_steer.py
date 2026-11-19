"""
Steering analysis on the teacher — the ground-truth version of the question.

Channel sets are the teacher's own two input pathways:
    value  — W_in[:, 0], how the plan is loaded
    go     — W_in[:, 1], how the movement is triggered
    both   — the two together

Windows: load (positive control) and delay (the question).
"""

import json
import numpy as np
import torch

from teacher import load_teacher, cued_response
import steer as S
import control as C


def slow_eigs(net, h):
    W = net.W_rec.detach().numpy(); b = net.b.detach().numpy()
    f = np.tanh(W @ h + b); D = 1 - f ** 2
    A = D[:, None] * W
    ev, V = np.linalg.eig(A); i = np.argmax(np.abs(ev))
    evl, Vl = np.linalg.eig(A.T); j = np.argmin(np.abs(evl - ev[i]))
    r = np.real(V[:, i]); r /= np.linalg.norm(r)
    l = np.real(Vl[:, j]); l /= np.linalg.norm(l)
    return r, l, D, float(abs(ev[i]))


def ang(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(min(1.0, abs(a @ b)))))


def main(path="runs/teacher_g0.9_n0.05_s0.pt", n=200, seed=21):
    torch.set_num_threads(1)
    net, _ = load_teacher(path)
    rng = np.random.default_rng(seed)
    u, z, m, meta = cued_response(n, rng=rng, return_meta=True)
    tl, tg, v = meta["t_load"], meta["t_go"], meta["v"]
    s, amp = S.teacher_sensitivity(net, u, tg)
    Win = net.W_in.detach().numpy()
    sets = {"value": Win[:, :1], "go": Win[:, 1:2], "both": Win}

    res = {"n_trials": n, "amp_vs_v_r": float(np.corrcoef(amp, v)[0, 1])}
    for kind in ("load", "delay"):
        idx = S.window_idx(kind, tl, tg)
        Qs = S.Q_from(s, idx)
        res[kind] = S.compare(Qs, sets)
        res[kind]["linearity"] = S.linearity_check(net, u, tg, s, idx)

    # geometry at the delay: where does the effective input direction point?
    with torch.no_grad():
        from teacher import run_noisy
        _, H = run_noisy(net, u, 0.0)
    H = H.numpy()
    rows = []
    for i in range(0, n, 5):
        t = tg[i] - 15                               # middle of the delay window
        r, l, D, lam = slow_eigs(net, H[i, t - 1])
        st = s[i, t]
        rows.append(dict(
            s_vs_right=ang(st, r),                   # input direction vs manifold tangent
            s_vs_Dleft=ang(st, D * l),               # vs gain-weighted left eigenvector
            right_vs_left=ang(r, l),                 # non-normality of the slow mode
            lam=lam,
            e_slow_s=float((st @ r) ** 2 / (st @ st))))  # causal analogue of the 0.352
    res["geometry"] = {k: float(np.median([r_[k] for r_ in rows])) for k in rows[0]}
    return res


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/teacher_g0.9_n0.05_s0.pt"
    outp = sys.argv[2] if len(sys.argv) > 2 else "runs/teacher_steer.json"
    res = main(path)
    print(json.dumps(res, indent=1))
    with open(outp, "w") as f:
        json.dump(res, f, indent=1)
