"""
Causal test: design a push from the fitted model, apply it to the true circuit.

Everything before this judged the student's channels INSIDE the student. Here the
push is taken out of the model and applied to the teacher, which is what a
stimulation experiment would do.

Bridge between the two state spaces: the recorded neurons. A push x added to the
student's pre-activation at time t changes the recorded log-rates at once by

    d logr = M_s(t) x,     M_s(t) = Dout F diag(1 - g_t^2)

and a push y added to the teacher's pre-activation changes them by

    d logr = M_T(t) y,     M_T(t) = diag(mod_sd / sd) C diag(1 - h_t^2)

The teacher push that corresponds to a student push is the smallest y with the
same immediate effect on the recorded neurons:  y = pinv(M_T) M_s x.

Measured, all in the delay window, on the true teacher:
  1. fraction of the teacher's full authority reached by (mapped) student channels,
     the student's own best direction, random student directions; against the
     teacher's true best direction and the ceiling for any pattern over the
     recorded neurons.
  2. actual change in the teacher's pull against the change the student predicts,
     for a push along the student's best direction and for the best use of its
     inferred channels. Run on teacher A (internal attractor) and teacher B (plan
     held by a sustained input).
  3. agreement between seeds, in recorded-neuron space, of the best direction and
     of the inferred-channel subspace.

    python causal.py
"""
import glob
import json
import re
import sys
import numpy as np
import torch

from teacher import load_teacher
import steer as S
import control as C
from student_steer import load_student, lever_decoder

MOD_SD = 0.7
NTR = 64                      # trials used
TARGET = 0.05                 # requested change in pull (inside the linear range)


L_RESP = 10                   # steps of response used to match pushes
RCOND = 1e-2


def teacher_resp(net, data, trials, idx, u):
    """R_T[(i,t)]: (nin*L, 100) linear response of recorded log-rates over the next L steps
    to a unit pre-activation push at time t, along the trial's own trajectory."""
    H = data["H_teacher"]; Cm = data["C"]; nin = data["n_heldin"]
    X = H.reshape(-1, H.shape[-1]) @ Cm.T
    sd = X.std(0)[:nin]
    Cs = (MOD_SD / sd)[:, None] * Cm[:nin]
    W = net.W_rec.detach().numpy()
    T = H.shape[1]; out = {}
    for i in trials:
        D = 1 - H[i] ** 2                                   # (T, 100)
        for t in idx[i]:
            M = np.diag(D[t]); rows = [Cs @ M]
            for j in range(1, L_RESP):
                if t + j >= T: break
                M = (D[t + j][:, None] * W) @ M
                rows.append(Cs @ M)
            out[(i, t)] = np.concatenate(rows, 0)
    return out


def student_resp(st, G, trials, idx):
    DF = (st.Dout.detach() @ st.F.detach()).numpy()
    W = st.W.detach().numpy(); Gn = G.numpy(); T = Gn.shape[1]; out = {}
    for i in trials:
        D = 1 - Gn[i] ** 2
        for t in idx[i]:
            M = np.diag(D[t]); rows = [DF @ M]
            for j in range(1, L_RESP):
                if t + j >= T: break
                M = (D[t + j][:, None] * W) @ M
                rows.append(DF @ M)
            out[(i, t)] = np.concatenate(rows, 0)
    return out


@torch.no_grad()
def teacher_amp(net, u, t_go, x):
    return S.teacher_amp_with_input(net, u, t_go, x)


def analyse(tag, teacher_path, data, paths, out):
    net, _ = load_teacher(teacher_path)
    tl, tg = data["t_load"], data["t_go"]
    trials = list(range(NTR))
    idx = S.window_idx("delay", tl, tg)
    u = torch.from_numpy(data["u_teacher"])
    sT, ampT = S.teacher_sensitivity(net, u, tg)           # (n, T, 100)
    RT = teacher_resp(net, data, trials, idx, u)
    pRT = {k: np.linalg.pinv(M, rcond=RCOND) for k, M in RT.items()}
    rng = np.random.default_rng(0)
    full = {i: float((sT[i, idx[i]] ** 2).sum()) for i in trials}
    QT = {i: sT[i, idx[i]].T @ sT[i, idx[i]] for i in trials}
    best1 = [float(np.linalg.eigvalsh(QT[i])[-1]) / full[i] for i in trials]
    ref = dict(best1_teacher=float(np.median(best1)), sens_abs=float(np.median([full[i] for i in trials])),
               r_amp_v=float(np.corrcoef(ampT, data["v"])[0, 1]))
    out[tag] = dict(teacher=ref, fits=[])
    print(tag, "teacher refs", ref, flush=True)
    base = teacher_amp(net, u[:NTR], tg[:NTR], torch.zeros(NTR, u.shape[1], net.n_hid))

    est = {}
    for p in paths:
        st, ck = load_student(p, data)
        ev = ck["eval"]
        if not np.isfinite(ev["bps_heldin"]) or ev["bps_heldin"] < 0:
            print("skip", p); continue
        w, b, r2 = lever_decoder(st, data, 1e-2)
        sS, ampS = S.student_sensitivity(st, w, b, tg)     # (n, T, N)
        with torch.no_grad():
            G, _ = st.generate(use_u=True)
        RS = student_resp(st, G, trials, idx)
        s2t = {k: pRT[k] @ RS[k] for k in RS}               # teacher push that mimics a student push
        t2s = {k: np.linalg.pinv(RS[k], rcond=RCOND) @ RT[k] for k in RS}   # student push that mimics a teacher push
        Bfit = st.B().detach().numpy(); k = Bfit.shape[1]; Pfit = C.orth(Bfit)
        QS = [sS[i, idx[i]].T @ sS[i, idx[i]] for i in trials]
        in_fit = float(np.median([np.trace(Pfit.T @ Q @ Pfit) / np.trace(Q) for Q in QS]))

        # the model's estimate of the true sensitivity
        shat = {i: np.stack([t2s[(i, t)].T @ sS[i, t] for t in idx[i]]) for i in trials}     # (L, 100)
        Qhat = sum(shat[i].T @ shat[i] / np.trace(shat[i].T @ shat[i]) for i in trials)
        evh, Vh = np.linalg.eigh(Qhat)
        d1 = Vh[:, -1:]; dk = Vh[:, -k:]
        des1 = float(np.median([float(d1[:, 0] @ QT[i] @ d1[:, 0]) / full[i] for i in trials]))
        desk = float(np.median([np.trace(dk.T @ QT[i] @ dk) / full[i] for i in trials]))
        des_tv = float(np.median([float((((shat[i] * sT[i, idx[i]]).sum(1)) ** 2 / (shat[i] ** 2).sum(1)).sum()) / full[i]
                                  for i in trials]))

        def mapped_frac(Bs):
            fr = []
            for i in trials:
                g = 0.0
                for t in idx[i]:
                    P = C.orth(s2t[(i, t)] @ Bs)
                    g += float(((P.T @ sT[i, t]) ** 2).sum())
                fr.append(g / full[i])
            return float(np.median(fr))

        f_fit = mapped_frac(Bfit)
        f_randk = float(np.mean([mapped_frac(C.orth(rng.standard_normal((st.N, k)))) for _ in range(5)]))

        # predicted against actual change in the pull
        def run(xT, pred):
            act = teacher_amp(net, u[:NTR], tg[:NTR], torch.from_numpy(xT)) - base
            E = (xT ** 2).sum((1, 2))
            eff = act ** 2 / (E * np.array([full[i] for i in trials]))
            return dict(ratio=float(np.median(act / pred)), ratio_q=[float(np.percentile(act / pred, q)) for q in (25, 75)],
                        actual=float(np.median(act)), eff=float(np.median(eff)))
        xT = np.zeros((NTR, u.shape[1], net.n_hid), np.float32)
        for i in trials:
            c = TARGET / float((shat[i] ** 2).sum())
            xT[i, idx[i]] = c * shat[i]
        r_design = run(xT, np.full(NTR, TARGET))
        xT = np.zeros((NTR, u.shape[1], net.n_hid), np.float32)
        for i in trials:
            a = sS[i, idx[i]] @ Pfit
            c = TARGET / float((a ** 2).sum())
            xs = c * a @ Pfit.T
            for j, t in enumerate(idx[i]):
                xT[i, t] = s2t[(i, t)] @ xs[j]
        r_chan = run(xT, np.full(NTR, TARGET))

        row = dict(path=p, lam=ck["lam_u"], seed=ck["seed"], cosmooth=ev["bps_cosmooth"], lever_r2=r2,
                   inside_fit=in_fit, teacher_fit=f_fit, teacher_randk=f_randk,
                   design1=des1, designk=desk, design_tv=des_tv, push_design=r_design, push_channels=r_chan)
        out[tag]["fits"].append(row)
        print(f"{tag} lam {row['lam']:g} s{row['seed']}: channels inside {in_fit:.2f} -> in teacher {f_fit:.2f} (rand {f_randk:.2f})"
              f" | designed push 1ch {des1:.2f} {k}ch {desk:.2f} tv {des_tv:.2f}"
              f" | actual/predicted designed {r_design['ratio']:.2f} (eff {r_design['eff']:.2f}) channels {r_chan['ratio']:.2f} (eff {r_chan['eff']:.3f})",
              flush=True)
        est[(ck["lam_u"], ck["seed"])] = (shat, {i: [C.orth(s2t[(i, t)] @ Bfit) for t in idx[i][::6]] for i in trials[:32]})

    agree = []
    keys = sorted(est)
    def c2(a, b):
        return float(np.mean((a * b).sum(1) ** 2 / ((a ** 2).sum(1) * (b ** 2).sum(1))))
    for a in range(len(keys)):
        for b_ in range(a + 1, len(keys)):
            if keys[a][0] != keys[b_][0]:
                continue
            s1, p1 = est[keys[a]]; s2, p2 = est[keys[b_]]
            cb = np.mean([c2(s1[i], s2[i]) for i in trials])
            cf = np.mean([np.linalg.norm(x.T @ y) ** 2 / min(x.shape[1], y.shape[1]) for i in p1 for x, y in zip(p1[i], p2[i])])
            agree.append(dict(lam=keys[a][0], seeds=[keys[a][1], keys[b_][1]], sens_cos2=float(cb), channels_overlap=float(cf)))
    out[tag]["agreement"] = agree
    out[tag]["truth_cos2"] = [dict(lam=kk[0], seed=kk[1], cos2=float(np.mean([c2(est[kk][0][i], sT[i, idx[i]]) for i in trials])))
                              for kk in keys]


if __name__ == "__main__":
    torch.set_num_threads(1)
    which = sys.argv[1] if len(sys.argv) > 1 else "AB"
    out = {}
    if "A" in which:
        dA = torch.load("runs/dataset.pt", weights_only=False)
        pa = [p for p in sorted(glob.glob("runs/student_k4_lam*_s?.pt")) if re.search(r"lam(0\.0001|0\.01|0\.1|1\.0)_s\d\.pt$", p)]
        analyse("A", "runs/teacher_g0.9_n0.05_s0.pt", dA, pa, out)
    if "B" in which:
        dB = torch.load("runs/dataset_B.pt", weights_only=False)
        pb = [p for p in sorted(glob.glob("runs/student_k4_lam*_s?_B.pt")) if re.search(r"lam(0\.0001|0\.01|0\.1|1\.0)_s\d_B\.pt$", p)]
        analyse("B", "runs/teacherB_sustained_s0.pt", dB, pb, out)
    if "C" in which:
        dC = torch.load("runs/dataset_C.pt", weights_only=False)
        pc = sorted(glob.glob("runs/student_k4_lam*_s?_C.pt"))
        analyse("C", "runs/teacherC_drift_s0.pt", dC, pc, out)
    json.dump(out, open(f"runs/causal_{which}.json", "w"), indent=1)
