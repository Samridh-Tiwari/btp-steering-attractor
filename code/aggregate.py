"""Collect every number the report quotes into runs/summary.json, from the saved results."""
import glob
import json
import re
import numpy as np

RIDGE = "0.01"


def students():
    out = []
    for p in sorted(glob.glob("runs/student_k*_lam*_s*_steer.json")):
        m = re.search(r"student_k(\d+)_lam([0-9.e-]+)_s(\d+)(_t2)?_steer", p)
        r = json.load(open(p))
        br = r["by_ridge"][RIDGE]
        d, l = br["delay"]["fit"], br["load"]["fit"]
        out.append(dict(
            k=int(m.group(1)), lam=float(m.group(2)), seed=int(m.group(3)),
            teacher=2 if m.group(4) else 1,
            bps=r["eval"]["bps_heldin"], cosmooth=r["eval"]["bps_cosmooth"],
            autonomy=r["eval"]["autonomy"], u_power=r["eval"]["u_power"],
            slow_lam=r["slow_eigs"][0], lever_r2=br["lever_r2"],
            fid_full=br["fidelity"]["r_full"], fid_silenced=br["fidelity"]["r_silenced"],
            delay_frac=d["frac_full"]["median"], delay_opt=d["frac_full_opt"]["median"],
            delay_aim=d["aim"]["median"], delay_xrand=d["ratio_rand"]["median"],
            delay_pct=d["pct"]["median"],
            delay_err=d["err_red"]["median"], delay_err_rand=d["err_red_rand"]["median"],
            delay_err_opt=d["err_red_opt"]["median"],
            load_frac=l["frac_full"]["median"], load_opt=l["frac_full_opt"]["median"],
            load_aim=l["aim"]["median"], load_xrand=l["ratio_rand"]["median"],
            ridge_spread_delay_aim=float(np.ptp([r["by_ridge"][k]["delay"]["fit"]["aim"]["median"]
                                                 for k in r["by_ridge"]])),
            random_frac=int(m.group(1)) / 64))
    return out


def group(rows, keys=("k", "lam", "teacher")):
    g = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    res = []
    for key, rr in sorted(g.items()):
        e = dict(zip(keys, key), n=len(rr))
        for f in rr[0]:
            if f in keys or f == "seed":
                continue
            vals = np.array([x[f] for x in rr], float)
            e[f] = float(vals.mean()); e[f + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else None
        res.append(e)
    return res


if __name__ == "__main__":
    S = students()
    T = json.load(open("runs/teacher_steer.json"))
    T2 = json.load(open("runs/teacher2_steer.json")) if glob.glob("runs/teacher2_steer.json") else None
    diag = json.load(open("runs/teacher_g0.9_n0.05_s0_diag.json"))
    figs = json.load(open("runs/figs_teacher.json"))
    out = dict(students=S, groups=group(S), teacher_steer=T, teacher2_steer=T2,
               teacher_diag=diag, decoder_pitfall=figs["fig2"])
    json.dump(out, open("runs/summary.json", "w"), indent=1)
    print(f"{len(S)} students")
    hdr = ("k", "lam", "teacher", "n", "cosmooth", "autonomy", "fid_silenced", "delay_frac",
           "delay_opt", "delay_xrand", "load_frac", "load_xrand")
    print(" ".join(f"{h:>11s}" for h in hdr))
    for e in out["groups"]:
        print(" ".join(f"{e[h]:11.3g}" if not isinstance(e[h], int) else f"{e[h]:11d}" for h in hdr))
