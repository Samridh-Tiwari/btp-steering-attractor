"""
Steering analysis on a fitted student — the SURF question, with ground truth.

For each student:
  1. behaviour = the student's predicted lever: a ridge decoder from generator
     state to the true lever, fit over the whole trial (the lever varies a lot
     during the pull, so this decoder is well conditioned — and its ridge is
     swept anyway).
  2. causal sensitivity of the predicted pull amplitude to an additive input,
     through the student's generator with its fitted g0 and u.
  3. compare the student's OWN inferred channels (span of B) against random
     and optimal subspaces of the same size, in the delay window and the load
     window.

Plus the ground-truth checks the SURF analysis could not do:
  * delay-silenced fidelity: zero the controller only during the delay and
    ask whether the student still produces the right pull amplitude. If the
    plan lives in the student's dynamics, nothing changes. If the controller
    was carrying it, the pull stops tracking v.
  * the student's slowest eigenvalue at delay states, against the teacher's.
"""

import glob
import json
import sys
import numpy as np
import torch

from lfads_lite import Student
import steer as S


def load_student(path, data):
    ck = torch.load(path, weights_only=False)
    n, T, _ = data["spikes"].shape
    nin = data["n_heldin"]
    st = Student(n, T, nin, N=ck["N"], k=ck["k"])
    st.load_state_dict(ck["state_dict"]); st.eval()
    return st, ck


@torch.no_grad()
def lever_decoder(st, data, ridge):
    G, _ = st.generate(use_u=True)
    X = G.reshape(-1, st.N).numpy()
    y = data["z"][..., 0].reshape(-1)
    Xm = X.mean(0); ym = y.mean()
    Xc = X - Xm
    w = np.linalg.solve(Xc.T @ Xc + ridge * len(X) * np.eye(st.N), Xc.T @ (y - ym))
    b = ym - Xm @ w
    r2 = 1 - ((Xc @ w - (y - ym)) ** 2).sum() / ((y - ym) ** 2).sum()
    return torch.tensor(w, dtype=torch.float32), float(b), float(r2)


@torch.no_grad()
def delay_silenced_fidelity(st, data, w, b):
    """Correlation of predicted pull amplitude with v: full model vs u zeroed in the delay."""
    tl, tg, v = data["t_load"], data["t_go"], data["v"]
    T = st.T
    mask = S.plateau_mask(tg, T)
    def amp(U):
        G, _ = st.generate(use_u=True, U=U)
        return ((G @ w + b) * mask).sum(1).numpy()
    a_full = amp(st.U.detach())
    U0 = st.U.detach().clone()
    for i in range(len(tg)):
        U0[i, tl[i] + 8:tg[i] - 3] = 0.0
    a_sil = amp(U0)
    return dict(r_full=float(np.corrcoef(a_full, v)[0, 1]),
                r_silenced=float(np.corrcoef(a_sil, v)[0, 1]),
                slope_full=float(np.polyfit(v, a_full, 1)[0]),
                slope_silenced=float(np.polyfit(v, a_sil, 1)[0]))


@torch.no_grad()
def slow_eig_student(st, data, n_pts=40):
    G, P = st.generate(use_u=True)
    W = st.W.detach().numpy()
    tg = data["t_go"]
    lams = []
    for i in np.linspace(0, len(tg) - 1, n_pts).astype(int):
        t = tg[i] - 15
        f = np.tanh(P[i, t].numpy())
        A = (1 - f ** 2)[:, None] * W
        lams.append(np.sort(np.abs(np.linalg.eigvals(A)))[::-1][:3])
    return np.median(lams, 0).tolist()


def analyse_student(path, data, ridges=(1e-4, 1e-2, 1.0)):
    torch.set_num_threads(1)
    st, ck = load_student(path, data)
    out = dict(path=path, k=ck["k"], lam_u=ck["lam_u"], eval={k: v for k, v in ck["eval"].items()
                                                              if not isinstance(v, list)})
    Bfit = st.B().detach().numpy()
    tl, tg = data["t_load"], data["t_go"]
    out["by_ridge"] = {}
    for ridge in ridges:
        w, b, r2 = lever_decoder(st, data, ridge)
        s, amp = S.student_sensitivity(st, w, b, tg)
        res = dict(lever_r2=r2, r_amp_v=float(np.corrcoef(amp, data["v"])[0, 1]))
        for kind in ("load", "delay"):
            Qs = S.Q_from(s, S.window_idx(kind, tl, tg))
            res[kind] = S.compare(Qs, {"fit": Bfit})
        res["fidelity"] = delay_silenced_fidelity(st, data, w, b)
        out["by_ridge"][str(ridge)] = res
    out["slow_eigs"] = slow_eig_student(st, data)
    return out


if __name__ == "__main__":
    data = torch.load("runs/dataset.pt", weights_only=False)
    paths = sys.argv[1:] or sorted(glob.glob("runs/student_k*_lam*_s*.pt"))
    for p in paths:
        r = analyse_student(p, data)
        outp = p.replace(".pt", "_steer.json")
        with open(outp, "w") as f:
            json.dump(r, f, indent=1)
        m = r["by_ridge"]["0.01"]
        print(f"{p}: autonomy {r['eval']['autonomy']:.3f}  delay aim {m['delay']['fit']['aim']['median']:.3f}"
              f"  x rand {m['delay']['fit']['ratio_rand']['median']:.2f}"
              f"  frac_full {m['delay']['fit']['frac_full']['median']:.3f}"
              f"  silenced r {m['fidelity']['r_silenced']:.3f}", flush=True)
