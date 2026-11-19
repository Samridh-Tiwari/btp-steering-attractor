"""Authority in absolute terms and by window. For each system: full authority (trace of Q)
in the delay window and in a window around the go cue, and the share reached by the
inferred channels / best-k / random in each. The delay-to-go ratio of full authority says
where in the trial the model's behaviour can be moved at all."""
import glob, json, re, sys
import numpy as np, torch
import steer as S, control as C
from teacher import load_teacher
from student_steer import load_student, lever_decoder

def windows(tl, tg):
    return dict(delay=[list(range(g - 30, g - 5)) for g in tg], go=[list(range(g - 3, g + 8)) for g in tg])

def main(tag, dp, tp, pat):
    data = torch.load(dp, weights_only=False); tl, tg = data["t_load"], data["t_go"]
    net, _ = load_teacher(tp); u = torch.from_numpy(data["u_teacher"])
    sT, _ = S.teacher_sensitivity(net, u, tg); W = windows(tl, tg)
    Win = net.W_in.detach().numpy()
    out = dict(teacher={}, fits=[])
    for name, idx in W.items():
        Qs = S.Q_from(sT, idx)
        r = S.compare(Qs, {"inputs": Win, "plan": Win[:, :1], "go": Win[:, 1:]}, n_random=50)
        out["teacher"][name] = dict(full=float(np.median([np.trace(Q) for Q in Qs])),
                                    inputs=r["inputs"]["frac_full"]["median"], plan=r["plan"]["frac_full"]["median"], go=r["go"]["frac_full"]["median"],
                                    best1=r["plan"]["frac_full_opt"]["median"])
    t = out["teacher"]; print(tag, "TEACHER full delay/go ratio %.3g | delay: inputs %.2f | go window: inputs %.2f plan %.2f go %.2f best1 %.2f" % (
        t["delay"]["full"] / t["go"]["full"], t["delay"]["inputs"], t["go"]["inputs"], t["go"]["plan"], t["go"]["go"], t["go"]["best1"]), flush=True)
    for p in sorted(glob.glob("runs/student_k4_lam*.pt")):
        if not re.match(pat, p): continue
        st, ck = load_student(p, data)
        if not np.isfinite(ck["eval"]["bps_heldin"]) or ck["eval"]["bps_heldin"] < 0: continue
        w, b, r2 = lever_decoder(st, data, 1e-2)
        sS, _ = S.student_sensitivity(st, w, b, tg); Bfit = st.B().detach().numpy()
        row = dict(lam=ck["lam_u"], seed=ck["seed"], cosmooth=ck["eval"]["bps_cosmooth"])
        for name, idx in W.items():
            Qs = S.Q_from(sS, idx); r = S.compare(Qs, {"fit": Bfit}, n_random=50)["fit"]
            row[name] = dict(full=float(np.median([np.trace(Q) for Q in Qs])), fit=r["frac_full"]["median"], best=r["frac_full_opt"]["median"],
                             rand=r["frac_full"]["median"] / r["ratio_rand"]["median"])
        out["fits"].append(row)
        print(f"{tag} lam {row['lam']:g} s{row['seed']}: full delay/go {row['delay']['full'] / row['go']['full']:.3g} | delay fit {row['delay']['fit']:.2f} best {row['delay']['best']:.2f}"
              f" | go fit {row['go']['fit']:.2f} best {row['go']['best']:.2f} rand {row['go']['rand']:.2f}", flush=True)
    return out

if __name__ == "__main__":
    torch.set_num_threads(1)
    res = dict(A=main("A", "runs/dataset.pt", "runs/teacher_g0.9_n0.05_s0.pt", r"runs/student_k4_lam(0\.0001|0\.01|0\.1|1\.0)_s\d\.pt$"),
               B=main("B", "runs/dataset_B.pt", "runs/teacherB_sustained_s0.pt", r"runs/student_k4_lam(0\.0001|0\.01|0\.1|1\.0)_s\d_B\.pt$"))
    json.dump(res, open("runs/abs_authority.json", "w"), indent=1)
