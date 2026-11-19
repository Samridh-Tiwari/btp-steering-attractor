"""How close are each model's single-trial rates to the true rates? (held-in neurons, all trials)"""
import glob, re, json
import numpy as np, torch
from student_steer import load_student
import event_student as E

def r2(pr, R):
    return float(1 - ((pr - R) ** 2).sum() / ((R - R.mean((0, 1))) ** 2).sum())

out = {}
torch.set_num_threads(1)
for tag, dp, suf in (("A", "runs/dataset.pt", ""), ("B", "runs/dataset_B.pt", "_B"), ("T2", "runs/dataset_T2.pt", "_T2"), ("An", "runs/dataset_An.pt", "_An")):
    data = torch.load(dp, weights_only=False); R = data["rates"][..., :50]
    rows = []
    for p in sorted(glob.glob("runs/student_k4_lam*_s*.pt")):
        if not re.match(rf"runs/student_k4_lam[0-9.e-]+_s\d{suf}\.pt$", p):
            continue
        st, ck = load_student(p, data)
        if not np.isfinite(ck["eval"]["bps_heldin"]) or ck["eval"]["bps_heldin"] < 0:
            continue
        with torch.no_grad():
            G, _ = st.generate(use_u=True); pr = torch.exp(st.logrates(G)).numpy()
        rows.append(dict(kind="free", path=p, lam=ck["lam_u"], seed=ck["seed"], rate_r2=r2(pr, R),
                         heldin=ck["eval"]["bps_heldin"], cosmooth=ck["eval"]["bps_cosmooth"]))
    for p in sorted(glob.glob(f"runs/event_s?{suf}.pt")):
        st, ck = E.load(p, data)
        with torch.no_grad():
            G, _ = st.generate(); pr = torch.exp(st.logrates(G)).numpy()
        rows.append(dict(kind="toldwhen", path=p, seed=ck["seed"], rate_r2=r2(pr, R),
                         heldin=ck["eval"]["bps_heldin"], cosmooth=ck["eval"]["bps_cosmooth"]))
    out[tag] = rows
    for kind in ("free", "toldwhen"):
        v = [r["rate_r2"] for r in rows if r["kind"] == kind]
        if v:
            print(tag, kind, len(v), "rate R2 median %.2f [%.2f, %.2f]" % (np.median(v), min(v), max(v)), flush=True)
    for lam in (1e-4, 0.01, 0.1, 1.0, 10.0):
        v = [r["rate_r2"] for r in rows if r["kind"] == "free" and abs(r["lam"] - lam) < 1e-12]
        if v:
            print("   lam", lam, np.round(v, 2))
json.dump(out, open("runs/rate_quality.json", "w"), indent=1)
