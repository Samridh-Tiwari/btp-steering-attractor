"""
Where does a fitted model keep the plan?  Three causal tests, run identically on the
teachers (truth) and on every student.

A fitted model can keep the plan in three places:
  (1) its own dynamics (a direction along which a displacement persists),
  (2) controller input fed in all through the delay,
  (3) controller input at or after the go cue (the plan is "re-injected" when needed).
The earlier silencing test zeroed the controller only inside the delay, so it could not
tell (1) from (3). These tests can.

  persistence : eigenvalues of the 20-step propagator along the trial's own trajectory
                (product of the Jacobians from go-30 to go-10). A direction that holds a
                displacement has |mu| near 1. Does not assume the state sits at a fixed point.
  swap        : keep trial i up to 8 steps after its plan is loaded, then give it the
                inputs of another trial j (aligned on the go cue). If the pull follows
                v_i the plan was in the state; if it follows v_j it was in the inputs.
                Variants: swap everything after the load ("all"), or only from 3 steps
                before the go cue ("go").
  shift       : 25 steps before the go cue, displace the state along the line of states
                by the equivalent of dv = 0.1, leave the inputs as they were, and measure
                how much of that shift reaches the pull (gain 1 = all of it).

    python plan_tests.py A B [C]
"""
import glob
import json
import re
import sys
import numpy as np
import torch

from teacher import load_teacher
import steer as S
from student_steer import load_student, lever_decoder

NSH = 0.1


# ---------------------------------------------------------------- teacher side
@torch.no_grad()
def t_run(net, u, h0=None, t0=0, H_prefix=None):
    """Run the teacher from t0 with state h0 (state after step t0-1). Returns lever (n,T)."""
    n, T, _ = u.shape
    h = torch.zeros(n, net.n_hid) if h0 is None else h0
    ys = torch.zeros(n, T)
    for t in range(t0, T):
        h = torch.tanh(h @ net.W_rec.T + u[:, t] @ net.W_in.T + net.b)
        ys[:, t] = h @ net.W_out[0] + net.b_out[0]
    return ys


def amp_of(y, tg):
    return (y * S.plateau_mask(tg, y.shape[1])).sum(1).numpy()


def two_slopes(amp, vi, vj):
    X = np.c_[vi, vj, np.ones(len(vi))]
    w = np.linalg.lstsq(X, amp, rcond=None)[0]
    return float(w[0]), float(w[1])


def swapped_inputs(U, tl, tg, perm, mode):
    """U (n,T,k) tensor. Returns U' where trial i gets trial perm[i]'s inputs, aligned on go."""
    n, T, k = U.shape
    out = U.clone()
    for i in range(n):
        j = perm[i]
        start = tl[i] + 8 if mode == "all" else tg[i] - 3
        for t in range(start, T):
            tj = t - tg[i] + tg[j]
            if mode == "all" and tj < tl[j] + 8:
                out[i, t] = 0.0                       # partner's delay has not begun
            elif 0 <= tj < T:
                out[i, t] = U[j, tj]
            else:
                out[i, t] = 0.0
    return out


def persistence(jac_seq):
    """jac_seq: list of Jacobians J_{t0+1}..J_{t1}. Returns |mu| sorted, and largest real positive mu."""
    Phi = np.eye(jac_seq[0].shape[0])
    for J in jac_seq:
        Phi = J @ Phi
    ev = np.linalg.eigvals(Phi)
    mag = np.sort(np.abs(ev))[::-1]
    realpos = [e.real for e in ev if abs(e.imag) < 1e-6 * max(1.0, abs(e)) and e.real > 0]
    return mag, (max(realpos) if realpos else 0.0)


def teacher_tests(path, data, sustained):
    net, _ = load_teacher(path)
    u = torch.from_numpy(data["u_teacher"]); tl, tg, v = data["t_load"], data["t_go"], data["v"]
    n = len(v); rng = np.random.default_rng(1); perm = rng.permutation(n)
    H = data["H_teacher"]
    y = t_run(net, u); a0 = amp_of(y, tg)
    slope = float(np.polyfit(v, a0, 1)[0])
    res = dict(slope=slope, r=float(np.corrcoef(a0, v)[0, 1]))
    for mode in ("all", "go"):
        us = swapped_inputs(u, tl, tg, perm, mode)
        a = amp_of(t_run(net, us), tg)
        si, sj = two_slopes(a, v, v[perm])
        res["swap_" + mode] = dict(own=si / slope, partner=sj / slope)
    # persistence
    W = net.W_rec.detach().numpy(); mags, rp = [], []
    for i in range(0, n, 8):
        m, r_ = persistence([(1 - H[i, t] ** 2)[:, None] * W for t in range(tg[i] - 29, tg[i] - 9)])
        mags.append(m[:6]); rp.append(r_)
    res["persist_top"] = np.median(mags, 0).tolist(); res["persist_realpos"] = float(np.median(rp))
    res["n_persist_05"] = float(np.median([(m > 0.5).sum() for m in mags]))
    # single-step eigenvalues: how many slow by magnitude
    cnt = []
    for i in range(0, n, 8):
        ev = np.abs(np.linalg.eigvals((1 - H[i, tg[i] - 15] ** 2)[:, None] * W))
        cnt.append([(ev > 0.9).sum(), (ev > 0.8).sum()])
    res["n_slow_09"], res["n_slow_08"] = [float(x) for x in np.median(cnt, 0)]
    # shift
    t0 = np.array(tg) - 25
    Hs = np.stack([H[i, t0[i] - 1] for i in range(n)])
    m = np.linalg.lstsq(np.c_[v, np.ones(n)], Hs, rcond=None)[0][0]
    a = np.zeros(n)
    for i in range(0, n, 2):
        h0 = torch.from_numpy((Hs[i] + NSH * m)[None].astype(np.float32))
        yy = t_run(net, u[i:i + 1], h0=h0, t0=int(t0[i]))
        a[i] = amp_of(yy, tg[i:i + 1])[0]
    sel = np.arange(0, n, 2)
    res["shift_gain"] = float(np.median((a[sel] - a0[sel]) / (NSH * slope)))
    return res


# ---------------------------------------------------------------- student side
@torch.no_grad()
def s_run(st, U, g0=None, t0=0, rows=None):
    Bm = st.B(); n = U.shape[0]
    g = (torch.tanh(st.g0) if rows is None else torch.tanh(st.g0[rows])) if g0 is None else g0
    G = torch.zeros(n, st.T, st.N)
    for t in range(t0, st.T):
        g = torch.tanh(g @ st.W.T + st.b + U[:, t] @ Bm.T)
        G[:, t] = g
    return G


def student_tests(path, data):
    st, ck = load_student(path, data)
    ev = ck["eval"]
    if not np.isfinite(ev["bps_heldin"]) or ev["bps_heldin"] < 0:
        return None
    tl, tg, v = data["t_load"], data["t_go"], data["v"]
    n = len(v); rng = np.random.default_rng(1); perm = rng.permutation(n)
    w, b, r2 = lever_decoder(st, data, 1e-2)
    mask = S.plateau_mask(tg, st.T)
    U = st.U.detach()
    with torch.no_grad():
        G, P = st.generate(use_u=True)
    amp = lambda GG: ((GG @ w + b) * mask).sum(1).numpy()
    a0 = amp(G); slope = float(np.polyfit(v, a0, 1)[0])
    res = dict(lam=ck["lam_u"], seed=ck["seed"], cosmooth=ev["bps_cosmooth"], heldin=ev["bps_heldin"], lever_r2=r2,
               slope=slope, r=float(np.corrcoef(a0, v)[0, 1]))
    for mode in ("all", "go"):
        a = amp(s_run(st, swapped_inputs(U, tl, tg, perm, mode)))
        si, sj = two_slopes(a, v, v[perm])
        res["swap_" + mode] = dict(own=si / slope, partner=sj / slope)
    # controller zeroed from 8 steps after the load to the end of the trial
    U0 = U.clone()
    for i in range(n):
        U0[i, tl[i] + 8:] = 0.0
    a = amp(s_run(st, U0))
    res["zero_all"] = dict(r=float(np.corrcoef(a, v)[0, 1]), slope=float(np.polyfit(v, a, 1)[0]) / slope)
    W = st.W.detach().numpy(); Pn = P.numpy(); mags, rp, cnt = [], [], []
    for i in range(0, n, 8):
        m, r_ = persistence([(1 - np.tanh(Pn[i, t]) ** 2)[:, None] * W for t in range(tg[i] - 29, tg[i] - 9)])
        mags.append(m[:6]); rp.append(r_)
        e1 = np.abs(np.linalg.eigvals((1 - np.tanh(Pn[i, tg[i] - 15]) ** 2)[:, None] * W))
        cnt.append([(e1 > 0.9).sum(), (e1 > 0.8).sum()])
    res["persist_top"] = np.median(mags, 0).tolist(); res["persist_realpos"] = float(np.median(rp))
    res["n_persist_05"] = float(np.median([(m > 0.5).sum() for m in mags]))
    res["n_slow_09"], res["n_slow_08"] = [float(x) for x in np.median(cnt, 0)]
    # shift along the line of states
    t0 = np.array(tg) - 25
    Gn = G.numpy()
    Gs = np.stack([Gn[i, t0[i] - 1] for i in range(n)])
    m = np.linalg.lstsq(np.c_[v, np.ones(n)], Gs, rcond=None)[0][0]
    g0 = torch.from_numpy((Gs + NSH * m).astype(np.float32)).clamp(-0.999, 0.999)
    # each trial starts at its own t0: run in groups of equal t0
    a = np.zeros(n)
    for tt in np.unique(t0):
        rows = np.where(t0 == tt)[0]
        GG = s_run(st, U[rows], g0=g0[rows], t0=int(tt))
        a[rows] = ((GG @ w + b) * mask[rows]).sum(1).numpy()
    res["shift_gain"] = float(np.median((a - a0) / (NSH * slope)))
    # how much of the controller's power sits in the delay, and at/after the go cue
    pw = (U.numpy() ** 2).sum(-1)
    d = np.zeros_like(pw, bool); g_ = np.zeros_like(pw, bool)
    for i in range(n):
        d[i, tl[i] + 8:tg[i] - 3] = True; g_[i, tg[i] - 3:] = True
    res["u_share_delay"] = float(pw[d].sum() / pw.sum()); res["u_share_go"] = float(pw[g_].sum() / pw.sum())
    return res


if __name__ == "__main__":
    torch.set_num_threads(1)
    cfg = dict(A=("runs/dataset.pt", "runs/teacher_g0.9_n0.05_s0.pt", False, r"runs/student_k4_lam[0-9.e-]+_s\d\.pt$"),
               B=("runs/dataset_B.pt", "runs/teacherB_sustained_s0.pt", True, r"runs/student_k4_lam[0-9.e-]+_s\d_B\.pt$"),
               T2=("runs/dataset_T2.pt", "runs/teacher_g0.9_n0.05_s1.pt", False, r"runs/student_k4_lam[0-9.e-]+_s\d_T2\.pt$"),
               C=("runs/dataset_C.pt", "runs/teacherC_drift_s0.pt", True, r"runs/student_k4_lam[0-9.e-]+_s\d_C\.pt$"))
    out = {}
    for tag in sys.argv[1:]:
        dp, tp, sus, pat = cfg[tag]
        data = torch.load(dp, weights_only=False)
        out[tag] = dict(teacher=teacher_tests(tp, data, sus), fits=[])
        t = out[tag]["teacher"]
        print(tag, "TEACHER swap_all own/partner %.2f/%.2f  swap_go %.2f/%.2f  persist %s realpos %.2f  n|mu|>.5 %g  n|lam|>.9 %g  shift gain %.2f"
              % (t["swap_all"]["own"], t["swap_all"]["partner"], t["swap_go"]["own"], t["swap_go"]["partner"],
                 np.round(t["persist_top"][:3], 2), t["persist_realpos"], t["n_persist_05"], t["n_slow_09"], t["shift_gain"]), flush=True)
        for p in sorted(glob.glob("runs/student_k4_lam*.pt")):
            if not re.match(pat, p):
                continue
            r = student_tests(p, data)
            if r is None:
                print("skip", p); continue
            out[tag]["fits"].append(r)
            print(f"{tag} lam {r['lam']:g} s{r['seed']} cs {r['cosmooth']:.3f}: swap_all own/partner {r['swap_all']['own']:.2f}/{r['swap_all']['partner']:.2f}"
                  f"  swap_go {r['swap_go']['own']:.2f}/{r['swap_go']['partner']:.2f}  zero_all slope {r['zero_all']['slope']:.2f}"
                  f"  persist {np.round(r['persist_top'][:2], 2)} realpos {r['persist_realpos']:.2f} n|mu|>.5 {r['n_persist_05']:g}"
                  f" n|lam|>.9 {r['n_slow_09']:g}  shift {r['shift_gain']:.2f}  u delay/go {r['u_share_delay']:.2f}/{r['u_share_go']:.2f}", flush=True)
        json.dump(out, open("runs/plan_tests_" + "".join(sys.argv[1:]) + ".json", "w"), indent=1)
