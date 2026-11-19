"""
Positive control for the steering metric, on the teacher, where the truth is known.

The teacher's own input channels have known jobs:
    value channel — loads the plan. By construction it can move the plan.
    go channel    — triggers the movement. It has no business moving the plan.

So for the PLAN readout (a decoder of v from the delay-period state), the
metric has to say: value channel >> random, go channel ~ random or below.
If it doesn't, the metric is broken and nothing downstream means anything.

Also measured here, because they bear directly on the SURF numbers:
  * the share of the plan readout lying in the slow subspace (the 0.352 analogue)
  * the angle between the slow mode's RIGHT eigenvector (the shape of the
    manifold — the direction you'd perturb ALONG) and its LEFT eigenvector
    (the direction an input must point to EXCITE that mode). In a non-normal
    network these differ, and "aim at the manifold" means aim at the left one.
  * how much of full actuation k optimally aimed channels recover.
"""

import json
import sys
import numpy as np
import torch

from teacher import load_teacher, cued_response, run_noisy
import control as C


@torch.no_grad()
def teacher_operating_points(net, n=400, seed=11):
    rng = np.random.default_rng(seed)
    u, z, m, meta = cued_response(n, rng=rng, return_meta=True)
    _, H = run_noisy(net, u, 0.0)
    H = H.numpy()
    v, tl, tg = meta["v"], meta["t_load"], meta["t_go"]
    # delay-period states (load finished + 10 .. go - 1) for the plan decoder
    Xs, ys = [], []
    for b in range(n):
        for t in range(tl[b] + 15, tg[b]):
            Xs.append(H[b, t]); ys.append(v[b])
    X, y = np.array(Xs), np.array(ys)
    Xc = np.c_[X, np.ones(len(X))]
    lam = 1e-3 * np.eye(Xc.shape[1]); lam[-1, -1] = 0
    w = np.linalg.solve(Xc.T @ Xc + lam, Xc.T @ y)
    c_plan = w[:-1]
    r2 = 1 - ((Xc @ w - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # operating points: mid-delay state of each trial
    tmid = ((tl + 5 + tg) // 2)
    ops = np.stack([H[b, tmid[b]] for b in range(n)])
    return ops, v, c_plan, r2


def jac_and_gain(net, h):
    W = net.W_rec.detach().numpy(); b = net.b.detach().numpy()
    f = np.tanh(W @ h + b)
    D = 1 - f ** 2
    return D[:, None] * W, D


def left_right_angle(A, m=1):
    """Angle between the slowest mode's right and left eigenvectors."""
    ev, V = np.linalg.eig(A)
    evl, Vl = np.linalg.eig(A.T)
    i = np.argmax(np.abs(ev)); j = np.argmin(np.abs(evl - ev[i]))
    r = np.real(V[:, i]); l = np.real(Vl[:, j])
    r /= np.linalg.norm(r); l /= np.linalg.norm(l)
    return float(np.degrees(np.arccos(min(1, abs(r @ l))))), float(abs(ev[i]))


def run(path, T_h=25, n_ops=40, k_list=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32)):
    net, cfg = load_teacher(path)
    ops, v, c_plan, r2 = teacher_operating_points(net)
    c_lever = net.W_out.detach().numpy()[0]
    Win = net.W_in.detach().numpy()
    sets = {"value": Win[:, :1], "go": Win[:, 1:2], "both": Win}
    rng = np.random.default_rng(0)
    pick = np.linspace(0, len(ops) - 1, n_ops).astype(int)
    acc = {"plan": [], "lever": []}
    extra = []
    for i in pick:
        A, D = jac_and_gain(net, ops[i])
        for name, c in (("plan", c_plan), ("lever", c_lever)):
            acc[name].append(C.analyse(A, D, c, sets, T=T_h, n_random=300, rng=rng,
                                       k_list=k_list))
        P_slow, mags = C.slow_subspace(A, 1)
        e_slow = float(np.sum((P_slow.T @ c_plan) ** 2) / np.sum(c_plan ** 2))
        ang, lam1 = left_right_angle(A)
        extra.append(dict(e_slow=e_slow, lr_angle=ang, lam1=lam1, lam2=float(mags[1])))

    def summarise(lst):
        out = {}
        for name in sets:
            rows = [r["sets"][name] for r in lst]
            out[name] = {key: float(np.median([r[key] for r in rows]))
                         for key in ("k", "G_fit", "G_rand_mean", "G_opt", "aim",
                                     "pct_of_random", "frac_full_fit", "frac_full_opt")}
        out["opt_curve"] = {k: float(np.median([r["opt_curve"][k] for r in lst])) for k in k_list}
        out["rand_curve"] = lst[0]["rand_curve"]
        return out

    res = dict(teacher=path, plan_decoder_r2=float(r2), T_h=T_h,
               plan=summarise(acc["plan"]), lever=summarise(acc["lever"]),
               slow=dict(e_slow_plan=float(np.median([e["e_slow"] for e in extra])),
                         lr_angle_deg=float(np.median([e["lr_angle"] for e in extra])),
                         lam1=float(np.median([e["lam1"] for e in extra])),
                         lam2=float(np.median([e["lam2"] for e in extra]))))
    return res


if __name__ == "__main__":
    res = run(sys.argv[1])
    print(json.dumps(res, indent=1))
    out = sys.argv[1].replace(".pt", "_control.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
