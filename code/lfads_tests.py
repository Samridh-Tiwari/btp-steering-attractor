"""
The same tests as plan_tests.py / fp_count.py / student_steer.py, run on full-LFADS fits.

Open loop  : generator alone, driven by the recorded controller outputs (posterior means).
             Replaying them reproduces the fitted trajectory exactly (checked).
Closed loop: the whole decoder, controller in the loop, fed the encoded spikes (ci).

  swap_all / swap_go : as before, on the controller outputs (open loop)
  swap_ci            : closed loop; trial i gets trial j's encoded spikes from 8 steps
                       after its load (aligned on go); the controller then has to decide
                       what to send. Tests the whole model rather than the generator.
  zero_all           : controller outputs zeroed from 8 steps after the load
  persistence        : 20-step propagator of the generator along the trial's own path
  shift (open/closed): displace the generator state 25 steps before go along the line
                       of states by dv = 0.1; open loop keeps the recorded inputs, closed
                       loop lets the controller see the displaced factors and respond
  fixed points       : autonomous generator (co = 0), marginal eigenvalue count
  authority          : share of full-state steering authority in the delay window held
                       by the controller's own push directions (dg/dco) vs random / best

    python lfads_tests.py runs/lfads_A_*.pt
"""
import glob
import json
import sys

import numpy as np
import torch

import lfads_full as L
import steer as S
import control as C
from plan_tests import swapped_inputs, two_slopes, persistence
from teacher import load_teacher

NSH = 0.1


def load(path):
    ck = torch.load(path, weights_only=False)
    hp = dict(ck["hps"])
    hps = L.make_hps(hp["encod_data_dim"], hp["encod_seq_len"], co_dim=hp["co_dim"],
                     gen_dim=hp["gen_dim"], con_dim=hp["con_dim"], fac_dim=hp["fac_dim"],
                     ic_dim=hp["ic_dim"], ic_enc_dim=hp["ic_enc_dim"], ci_enc_dim=hp["ci_enc_dim"],
                     dropout=hp["dropout_rate"], ic_win=hp.get("ic_enc_seq_len", 0))
    m = L.LFADS(hps); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def lever(G, z, ridge=1e-2):
    X = G.reshape(-1, G.shape[-1]).numpy().astype(np.float64)
    y = z[..., 0].reshape(-1)
    Xm, ym = X.mean(0), y.mean(); Xc = X - Xm
    w = np.linalg.solve(Xc.T @ Xc + ridge * len(X) * np.eye(X.shape[1]), Xc.T @ (y - ym))
    b = ym - Xm @ w
    r2 = 1 - ((Xc @ w - (y - ym)) ** 2).sum() / ((y - ym) ** 2).sum()
    return torch.tensor(w, dtype=torch.float32), float(b), float(r2)


def v_decode(Gn, v, ts, ridge=1e-2):
    """Cross-validated R^2 of the plan value v from the state at the given times (even/odd trials)."""
    X = np.stack([Gn[i, t] for i, t in enumerate(ts)]).astype(np.float64)
    tr = np.arange(len(v)) % 2 == 0
    mx = X[tr].mean(0); my = v[tr].mean(); Xc = X[tr] - mx
    w = np.linalg.solve(Xc.T @ Xc + ridge * len(Xc) * np.eye(X.shape[1]), Xc.T @ (v[tr] - my))
    p = (X[~tr] - mx) @ w + my
    return float(1 - ((p - v[~tr]) ** 2).sum() / ((v[~tr] - v[~tr].mean()) ** 2).sum())


def cell_jac_h(cell, u, h):
    return torch.autograd.functional.jacobian(lambda hh: cell(u[None], hh[None])[0], h).numpy()


def cell_jac_u(cell, u, h):
    return torch.autograd.functional.jacobian(lambda uu: cell(uu[None], h[None])[0], u).numpy()


def sensitivity(m, g0, U, w, b, tg):
    """s (n,T,N): d amp / d x(t) for x added to the generator state after step t (open loop)."""
    cell = m.gen_cell()
    n, T, _ = U.shape
    x = torch.zeros(n, T, m.hps.gen_dim, requires_grad=True)
    g = g0.detach(); ys = []
    for t in range(T):
        g = cell(U[:, t], g) + x[:, t]
        ys.append(g @ w + b)
    y = torch.stack(ys, 1)
    amp = (y * S.plateau_mask(tg, T)).sum(1)
    amp.sum().backward()
    return x.grad.detach().numpy()


def find_fp(cell, starts, k, steps=1500, lr=0.01):
    h = starts.clone().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=lr)
    u0 = torch.zeros(len(starts), k)
    for _ in range(steps):
        q = 0.5 * ((cell(u0, h) - h) ** 2).sum(1)
        opt.zero_grad(); q.sum().backward(); opt.step()
    with torch.no_grad():
        q = 0.5 * ((cell(u0, h) - h) ** 2).sum(1)
    return h.detach(), q.numpy()


def tests(path, data, do_fp=True, do_auth=True, rows=None):
    m, ck = load(path)
    if rows is not None:
        data = L.subset(data, np.asarray(rows))
    tl, tg, v = data["t_load"], data["t_go"], data["v"]
    n = len(v); T = data["spikes"].shape[1]
    rng = np.random.default_rng(1); perm = rng.permutation(n)
    o = L.infer(m, data)
    G, U, g0, ci, con = o["gen"], o["co_mean"], o["g0"], o["ci"], o["con"]
    w, b, r2 = lever(G, data["z"])
    mask = S.plateau_mask(tg, T)
    amp = lambda GG: ((GG @ w + b) * mask).sum(1).numpy()
    a0 = amp(G); slope = float(np.polyfit(v, a0, 1)[0])
    ev = ck["eval"]
    sfx = "" if rows is None or "bps_heldin_test" not in ev else ("_test" if len(rows) == len(ev["test_idx"]) else "_train")
    res = dict(path=path, kl_co=ck["kl_co"], seed=ck["seed"], n_trials=n, heldin=ev["bps_heldin" + sfx],
               cosmooth=ev["bps_cosmooth" + sfx], rate_r2=ev["rate_r2" + sfx], autonomy=ev["autonomy"],
               lever_r2=r2, slope=slope, r=float(np.corrcoef(a0, v)[0, 1]))
    res["co_std_median"] = float(o["co_std"].median())
    res["co_prior_std"] = float(torch.exp(0.5 * m.co_prior.lognvars).median() /
                                torch.sqrt(1 - torch.exp(-1.0 / torch.exp(m.co_prior.logtaus)) ** 2).median())
    # is the plan known before it is given? decode v from the state 3 steps before the load
    res["v_before_load"] = v_decode(G.numpy(), v, [t - 3 for t in tl])
    res["v_at_start"] = v_decode(G.numpy(), v, [2] * n)
    res["v_mid_delay"] = v_decode(G.numpy(), v, [t - 15 for t in tg])
    # replay checks
    res["replay_open"] = float((m.run_generator(g0, U) - G).abs().max())
    Gc, Uc = m.run_closed_loop(g0, ci)
    res["replay_closed"] = float((Gc - G).abs().max())
    # swaps, open loop
    for mode in ("all", "go"):
        a = amp(m.run_generator(g0, swapped_inputs(U, tl, tg, perm, mode)))
        si, sj = two_slopes(a, v, v[perm])
        res["swap_" + mode] = dict(own=si / slope, partner=sj / slope)
    # swap, closed loop (encoded spikes of the partner)
    Gs, Us = m.run_closed_loop(g0, swapped_inputs(ci, tl, tg, perm, "all"))
    si, sj = two_slopes(amp(Gs), v, v[perm])
    res["swap_ci"] = dict(own=si / slope, partner=sj / slope)
    # zero controller after the load
    U0 = U.clone()
    for i in range(n):
        U0[i, tl[i] + 8:] = 0.0
    a = amp(m.run_generator(g0, U0))
    res["zero_all"] = dict(r=float(np.corrcoef(a, v)[0, 1]), slope=float(np.polyfit(v, a, 1)[0]) / slope)
    # persistence along the trajectory
    cell = m.gen_cell(); mags, rp, cnt = [], [], []
    stp = max(1, n // 32)
    for i in range(0, n, stp):
        Js = [cell_jac_h(cell, U[i, t], G[i, t - 1]) for t in range(tg[i] - 29, tg[i] - 9)]
        mg, r_ = persistence(Js); mags.append(mg[:6]); rp.append(r_)
        e1 = np.abs(np.linalg.eigvals(cell_jac_h(cell, U[i, tg[i] - 15], G[i, tg[i] - 16])))
        cnt.append([(e1 > 0.9).sum(), (e1 > 0.8).sum()])
    res["persist_top"] = np.median(mags, 0).tolist(); res["persist_realpos"] = float(np.median(rp))
    res["n_persist_05"] = float(np.median([(mm > 0.5).sum() for mm in mags]))
    res["n_slow_09"], res["n_slow_08"] = [float(x) for x in np.median(cnt, 0)]
    # shift along the line of states, open and closed loop
    t0 = np.array(tg) - 25
    Gn = G.numpy()
    Gs0 = np.stack([Gn[i, t0[i] - 1] for i in range(n)])
    mv = np.linalg.lstsq(np.c_[v, np.ones(n)], Gs0, rcond=None)[0][0]
    gsh = torch.from_numpy((Gs0 + NSH * mv).astype(np.float32))
    a_o = np.zeros(n); a_c = np.zeros(n)
    for tt in np.unique(t0):
        rows = np.where(t0 == tt)[0]
        GG = m.run_generator(gsh[rows], U[rows], t0=int(tt))
        a_o[rows] = ((GG @ w + b) * mask[rows]).sum(1).numpy()
        GG, _ = m.run_closed_loop(None, ci[rows], t0=int(tt), gen_prev=gsh[rows], con_prev=con[rows, tt - 1])
        a_c[rows] = ((GG @ w + b) * mask[rows]).sum(1).numpy()
    res["shift_gain"] = float(np.median((a_o - a0) / (NSH * slope)))
    res["shift_gain_closed"] = float(np.median((a_c - a0) / (NSH * slope)))
    # controller power by epoch
    pw = (U.numpy() ** 2).sum(-1)
    d = np.zeros_like(pw, bool); g_ = np.zeros_like(pw, bool); ld = np.zeros_like(pw, bool)
    for i in range(n):
        ld[i, tl[i] - 2:tl[i] + 8] = True; d[i, tl[i] + 8:tg[i] - 3] = True; g_[i, tg[i] - 3:] = True
    res["u_share_load"] = float(pw[ld].sum() / pw.sum())
    res["u_share_delay"] = float(pw[d].sum() / pw.sum()); res["u_share_go"] = float(pw[g_].sum() / pw.sum())
    # fixed points of the autonomous generator
    if do_fp:
        sel = list(range(0, n, max(1, n // 32)))
        hfp, q = find_fp(cell, torch.stack([G[i, tg[i] - 15] for i in sel]), m.hps.co_dim)
        rows = []
        for j in range(len(sel)):
            if q[j] > 1e-5:
                continue
            e = np.abs(np.linalg.eigvals(cell_jac_h(cell, torch.zeros(m.hps.co_dim), hfp[j])))
            rows.append([(e > 0.9).sum(), ((e > 0.95) & (e < 1.05)).sum(), (e > 1.05).sum()])
        res["fp"] = dict(n_points=len(rows), q_median=float(np.median(q)))
        if rows:
            md = np.median(rows, 0)
            res["fp"].update(slow09=float(md[0]), marginal=float(md[1]), unstable=float(md[2]),
                             marginal_max=float(np.max(np.array(rows)[:, 1])))
    # authority in the delay window
    if do_auth:
        s = sensitivity(m, g0, U, w, b, tg)
        idx = S.window_idx("delay", tl, tg)
        rngr = np.random.default_rng(0); rows = []
        k = m.hps.co_dim
        for i in range(0, n, 2):
            Q = s[i, idx[i]].T @ s[i, idx[i]]
            Gfull = float(np.trace(Q))
            if Gfull <= 0:
                continue
            Ju = [cell_jac_u(cell, U[i, t], G[i, t - 1]) for t in idx[i]]
            Uu, _, _ = np.linalg.svd(np.concatenate(Ju, 1), full_matrices=False)
            P = Uu[:, :k]
            g = C.G_of(Q, P); gr = C.G_random(Q, k, 200, rngr)
            evq = np.linalg.eigvalsh((Q + Q.T) / 2)[::-1]
            # time-varying version: each step's own push directions
            gtv = sum(float(np.sum((C.orth(J).T @ s[i, t]) ** 2)) for J, t in zip(Ju, idx[i]))
            rows.append(dict(frac_full=g / Gfull, frac_rand=float(gr.mean()) / Gfull,
                             frac_opt=float(evq[:k].sum()) / Gfull, frac_opt1=float(evq[0]) / Gfull,
                             frac_tv=gtv / Gfull, pct=float((gr < g).mean() * 100),
                             err_red=C.error_reduction(g, Gfull / 9), err_red_opt=C.error_reduction(float(evq[:k].sum()), Gfull / 9)))
        res["authority"] = {key: float(np.median([r_[key] for r_ in rows])) for key in rows[0]}
        # the same in the load window (5 steps from the plan cue): can the channels write the plan?
        idl = S.window_idx("load", tl, tg); rl = []
        for i in range(0, n, 2):
            Q = s[i, idl[i]].T @ s[i, idl[i]]; Gfull = float(np.trace(Q))
            if Gfull <= 0:
                continue
            Ju = [cell_jac_u(cell, U[i, t], G[i, t - 1]) for t in idl[i]]
            P = np.linalg.svd(np.concatenate(Ju, 1), full_matrices=False)[0][:, :k]
            rl.append(dict(frac_full=C.G_of(Q, P) / Gfull, frac_rand=float(C.G_random(Q, k, 200, rngr).mean()) / Gfull))
        res["authority_load"] = {key: float(np.median([r_[key] for r_ in rl])) for key in rl[0]}
    # where to push in the true circuit (same bridge as event_tests, student push = state push)
    tp = TEACHERS.get(ck["data"])
    if tp is not None and do_auth:
        net, _ = load_teacher(tp)
        sT = teacher_sens(net, data, tg)
        cell_fac = m.decoder.rnn.cell.fac_linear
        Wf = torch.nn.functional.normalize(cell_fac.weight, p=2, dim=1).detach().numpy()
        DF = m.readout.weight.detach().numpy() @ Wf
        res["bridge"] = bridge_state(data, sT, s, DF, list(range(64)), S.window_idx("delay", tl, tg))
    return res


def teacher_sens(net, data, tg):
    """True push sensitivity of the teacher along the recorded path (with its own noise, if any)."""
    u = torch.from_numpy(data["u_teacher"])
    if "xi" not in data:
        return S.teacher_sensitivity(net, u, tg)[0]
    xi = torch.from_numpy(data["xi"])
    B, T, _ = u.shape
    x = torch.zeros(B, T, net.n_hid, requires_grad=True)
    h = torch.zeros(B, net.n_hid); ys = []
    for t in range(T):
        h = torch.tanh(h @ net.W_rec.T + u[:, t] @ net.W_in.T + net.b + xi[:, t] + x[:, t])
        ys.append(h @ net.W_out[0] + net.b_out[0])
    amp = (torch.stack(ys, 1) * S.plateau_mask(tg, T)).sum(1)
    amp.sum().backward()
    return x.grad.detach().numpy()


TEACHERS = {"runs/dataset.pt": "runs/teacher_g0.9_n0.05_s0.pt",
            "runs/dataset_An.pt": "runs/teacher_g0.9_n0.05_s0.pt",
            "runs/dataset_B.pt": "runs/teacherB_sustained_s0.pt",
            "runs/dataset_T2.pt": "runs/teacher_g0.9_n0.05_s1.pt"}


def bridge_state(data, sT, sS, DF, trials, idx, mod_sd=0.7):
    """As event_tests.bridge, but the student's push is added to its state, so the
    student's immediate effect on the recorded log-rates is DF (no tanh factor)."""
    H = data["H_teacher"]; Cm = data["C"]; nin = data["n_heldin"]
    X = H.reshape(-1, H.shape[-1]) @ Cm.T
    Cs = (mod_sd / X.std(0)[:nin])[:, None] * Cm[:nin]
    Msp = np.linalg.pinv(DF, rcond=1e-2)
    cos, frac, ceil = [], [], []
    for i in trials:
        num = 0.0; full = float((sT[i, idx[i]] ** 2).sum()); cl = 0.0
        for t in idx[i]:
            MT = Cs * (1 - H[i, t] ** 2)[None]
            sh = MT.T @ (Msp.T @ sS[i, t])
            c = float(sh @ sT[i, t] / (np.linalg.norm(sh) * np.linalg.norm(sT[i, t]) + 1e-30))
            cos.append(c)
            num += np.sign(c) * (sh @ sT[i, t]) ** 2 / (sh @ sh + 1e-30)
            P = C.orth(MT.T); cl += float(((P.T @ sT[i, t]) ** 2).sum())
        frac.append(num / full); ceil.append(cl / full)
    return dict(cos_median=float(np.median(cos)), cos_pos=float((np.array(cos) > 0).mean()),
                frac_teacher=float(np.median(frac)), ceiling=float(np.median(ceil)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    paths = sys.argv[1:]
    out_path = "runs/lfads_tests.json"
    try:
        allres = json.load(open(out_path))
    except Exception:
        allres = {}
    for p in paths:
        ck = torch.load(p, weights_only=False)
        data = torch.load(ck["data"], weights_only=False)
        ev = ck["eval"]
        if "test_idx" in ev:
            r = tests(p, data, rows=ev["test_idx"])
            r["train"] = tests(p, data, rows=ev["train_idx"], do_fp=False, do_auth=False)
            r["train"] = {k: r["train"][k] for k in ("heldin", "cosmooth", "rate_r2", "r", "v_before_load", "v_at_start",
                                                      "v_mid_delay", "swap_all", "swap_ci", "shift_gain", "u_share_go")}
        else:
            r = tests(p, data)
        allres[p] = r
        json.dump(allres, open(out_path, "w"), indent=1)
        print(p, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                             if k not in ("persist_top", "path")}), flush=True)
