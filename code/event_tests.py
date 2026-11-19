"""
Tests on the event students (inputs only allowed at the known cue times), and the
"where to push" test for both kinds of student.

For each event student, on teacher A data and on teacher B data:
  * fit: held-in and held-out bits per spike (to compare with the free students)
  * persistence spectrum, number of slow modes, shift gain  (same definitions as plan_tests)
  * steering inside the model: share of full authority of the best single direction
  * where to push in the TRUE circuit: the model's sensitivity is translated to the teacher
    through the recorded neurons (a teacher push y changes the recorded log-rates at once by
    M_T y; the model reads that as the smallest student push with the same immediate effect,
    x = pinv(M_s) M_T y; the model's predicted change in pull is s_S . x). This gives the
    model's estimate of the teacher's sensitivity, s_hat = M_T^T pinv(M_s)^T s_S, which is
    compared with the true s_T: direction (signed cosine), share of the teacher's full
    authority a push along s_hat reaches, and predicted / actual size.

    python event_tests.py
"""
import glob
import json
import re
import numpy as np
import torch

import steer as S
import control as C
import event_student as E
from teacher import load_teacher
from student_steer import load_student, lever_decoder
from plan_tests import persistence, t_run, amp_of, NSH

MOD_SD = 0.7
NTR = 64


def sens_event(st, w, b, tg):
    n = st.amp.shape[0]; T = st.T
    x = torch.zeros(n, T, st.N, requires_grad=True)
    dr = st.drive().detach(); W = st.W.detach(); bb = st.b.detach()
    g = torch.tanh(st.g0.detach()).expand(n, -1)
    ys = []
    for t in range(T):
        g = torch.tanh(g @ W.T + bb + dr[:, t] + x[:, t])
        ys.append(g @ w + b)
    amp = (torch.stack(ys, 1) * S.plateau_mask(tg, T)).sum(1)
    amp.sum().backward()
    return x.grad.detach().numpy(), amp.detach().numpy()


def bridge(data, net, sT, sS, G, DF, trials, idx):
    """Model's estimate of the teacher's push sensitivity, against the truth."""
    H = data["H_teacher"]; Cm = data["C"]; nin = data["n_heldin"]
    X = H.reshape(-1, H.shape[-1]) @ Cm.T
    Cs = (MOD_SD / X.std(0)[:nin])[:, None] * Cm[:nin]
    cos, frac, ratio, ceil = [], [], [], []
    for i in trials:
        num = 0.0; full = float((sT[i, idx[i]] ** 2).sum()); pr = 0.0; ac = 0.0; cl = 0.0
        for t in idx[i]:
            MT = Cs * (1 - H[i, t] ** 2)[None]
            Ms = DF * (1 - G[i, t] ** 2)[None]
            sh = MT.T @ (np.linalg.pinv(Ms, rcond=1e-2).T @ sS[i, t])
            c = float(sh @ sT[i, t] / (np.linalg.norm(sh) * np.linalg.norm(sT[i, t]) + 1e-30))
            cos.append(c)
            num += np.sign(c) * (sh @ sT[i, t]) ** 2 / (sh @ sh + 1e-30)
            pr += float(sh @ sh); ac += float(sh @ sT[i, t])
            P = C.orth(MT.T); cl += float(((P.T @ sT[i, t]) ** 2).sum())
        frac.append(num / full); ratio.append(pr / ac if abs(ac) > 1e-30 else np.nan); ceil.append(cl / full)
    return dict(cos_median=float(np.median(cos)), cos_pos=float((np.array(cos) > 0).mean()),
                frac_teacher=float(np.median(frac)), pred_over_actual=float(np.nanmedian(ratio)), ceiling=float(np.median(ceil)))


def run(tag, dp, tp):
    data = torch.load(dp, weights_only=False); tl, tg, v = data["t_load"], data["t_go"], data["v"]
    n = len(v); net, _ = load_teacher(tp); u = torch.from_numpy(data["u_teacher"])
    sT, ampT = S.teacher_sensitivity(net, u, tg)
    trials = list(range(NTR)); idx = S.window_idx("delay", tl, tg)
    out = dict(event=[], free=[])
    # event students
    for p in sorted(glob.glob("runs/event_s?%s.pt" % ("" if tag == "A" else "_" + tag))):
        st, ck = E.load(p, data)
        w, b, r2 = lever_decoder(st, data, 1e-2)
        with torch.no_grad():
            G, P = st.generate()
        mask = S.plateau_mask(tg, st.T)
        a0 = ((G @ w + b) * mask).sum(1).numpy(); slope = float(np.polyfit(v, a0, 1)[0])
        row = dict(path=p, seed=ck["seed"], heldin=ck["eval"]["bps_heldin"], cosmooth=ck["eval"]["bps_cosmooth"], lever_r2=r2,
                   r_amp_v=float(np.corrcoef(a0, v)[0, 1]), slope=slope)
        Wn = st.W.detach().numpy(); Pn = P.numpy(); mags, rp, cnt = [], [], []
        for i in range(0, n, 8):
            m, r_ = persistence([(1 - np.tanh(Pn[i, t]) ** 2)[:, None] * Wn for t in range(tg[i] - 29, tg[i] - 9)])
            mags.append(m[:6]); rp.append(r_)
            e1 = np.abs(np.linalg.eigvals((1 - np.tanh(Pn[i, tg[i] - 15]) ** 2)[:, None] * Wn))
            cnt.append([(e1 > 0.9).sum(), (e1 > 0.8).sum()])
        row["persist_top"] = np.median(mags, 0).tolist(); row["persist_realpos"] = float(np.median(rp))
        row["n_persist_05"] = float(np.median([(m > 0.5).sum() for m in mags]))
        row["n_slow_09"], row["n_slow_08"] = [float(x) for x in np.median(cnt, 0)]
        # shift
        t0 = np.array(tg) - 25; Gn = G.numpy()
        Gs = np.stack([Gn[i, t0[i] - 1] for i in range(n)])
        m = np.linalg.lstsq(np.c_[v, np.ones(n)], Gs, rcond=None)[0][0]
        g0 = torch.from_numpy((Gs + NSH * m).astype(np.float32)).clamp(-0.999, 0.999)
        a = np.zeros(n); dr = st.drive().detach()
        with torch.no_grad():
            for tt in np.unique(t0):
                rows = np.where(t0 == tt)[0]
                GG, _ = st.generate(drive=dr[rows], g0=g0[rows], t0=int(tt))
                a[rows] = ((GG @ w + b) * mask[rows]).sum(1).numpy()
        row["shift_gain"] = float(np.median((a - a0) / (NSH * slope))) if abs(slope) > 1e-6 else float("nan")
        # steering inside the model
        sS, _ = sens_event(st, w, b, tg)
        Qs = S.Q_from(sS, idx)
        row["best1_inside"] = float(np.median([np.linalg.eigvalsh(Q)[-1] / np.trace(Q) for Q in Qs if np.trace(Q) > 0]))
        DF = (st.Dout.detach() @ st.F.detach()).numpy()
        row["bridge"] = bridge(data, net, sT, sS, Gn, DF, trials, idx)
        out["event"].append(row)
        br = row["bridge"]
        print(f"{tag} event s{row['seed']}: bps in {row['heldin']:.3f} out {row['cosmooth']:.3f} lever R2 {r2:.2f} r(amp,v) {row['r_amp_v']:.2f}"
              f" | persist {np.round(row['persist_top'][:2], 2)} realpos {row['persist_realpos']:.2f} n|lam|>.8 {row['n_slow_08']:g} shift {row['shift_gain']:.2f}"
              f" best1 {row['best1_inside']:.2f} | bridge cos {br['cos_median']:.2f} pos {br['cos_pos']:.2f} frac {br['frac_teacher']:.2f}"
              f" pred/actual {br['pred_over_actual']:.2f} (ceiling {br['ceiling']:.2f})", flush=True)
    # free students, bridge only
    pat = r"runs/student_k4_lam(0\.0001|0\.01|0\.1|1\.0)_s\d%s\.pt$" % ("" if tag == "A" else "_" + tag)
    for p in sorted(glob.glob("runs/student_k4_lam*.pt")):
        if not re.match(pat, p):
            continue
        st, ck = load_student(p, data)
        if not np.isfinite(ck["eval"]["bps_heldin"]) or ck["eval"]["bps_heldin"] < 0:
            continue
        w, b, r2 = lever_decoder(st, data, 1e-2)
        sS, _ = S.student_sensitivity(st, w, b, tg)
        with torch.no_grad():
            G, _ = st.generate(use_u=True)
        DF = (st.Dout.detach() @ st.F.detach()).numpy()
        br = bridge(data, net, sT, sS, G.numpy(), DF, trials, idx)
        out["free"].append(dict(lam=ck["lam_u"], seed=ck["seed"], cosmooth=ck["eval"]["bps_cosmooth"], heldin=ck["eval"]["bps_heldin"], bridge=br))
        print(f"{tag} free lam {ck['lam_u']:g} s{ck['seed']}: bridge cos {br['cos_median']:.2f} pos {br['cos_pos']:.2f} frac {br['frac_teacher']:.2f} pred/actual {br['pred_over_actual']:.2f}", flush=True)
    return out


if __name__ == "__main__":
    torch.set_num_threads(1)
    res = dict(A=run("A", "runs/dataset.pt", "runs/teacher_g0.9_n0.05_s0.pt"),
               B=run("B", "runs/dataset_B.pt", "runs/teacherB_sustained_s0.pt"))
    json.dump(res, open("runs/event_tests.json", "w"), indent=1)
