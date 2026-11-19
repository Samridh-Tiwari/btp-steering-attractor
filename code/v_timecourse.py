"""When does each model 'know' the plan? Cross-validated R^2 of v from the state at each
time, aligned on the plan cue (t_load). For split LFADS fits, held-out and training trials
are done separately."""
import glob, json, sys
import numpy as np, torch
import lfads_full as L, lfads_tests as LT
from student_steer import load_student

LAGS = list(range(-9, 31))

def curve(Gn, v, tl, rows):
    out = []
    for d in LAGS:
        ts = [tl[i] + d for i in rows]
        out.append(LT.v_decode(Gn[rows], v[rows], ts))
    return out

if __name__ == "__main__":
    torch.set_num_threads(1)
    data = torch.load("runs/dataset.pt", weights_only=False)
    v, tl = data["v"], np.array(data["t_load"]); n = len(v); allr = np.arange(n)
    res = dict(lags=LAGS)
    res["teacher"] = curve(data["H_teacher"], v, tl, allr)
    lite = []
    for p in sorted(glob.glob("runs/student_k4_lam0.1_s?.pt")):
        st, ck = load_student(p, data)
        with torch.no_grad():
            G, _ = st.generate(use_u=True)
        lite.append(curve(G.numpy(), v, tl, allr))
    res["lite"] = lite
    for p in sys.argv[1:]:
        m, ck = LT.load(p); o = L.infer(m, data); Gn = o["gen"].numpy()
        ev = ck["eval"]
        if "test_idx" in ev:
            res[p] = dict(test=curve(Gn, v, tl, np.array(ev["test_idx"])), train=curve(Gn, v, tl, np.array(ev["train_idx"])))
        else:
            res[p] = dict(all=curve(Gn, v, tl, allr))
        print(p, "done", flush=True)
    json.dump(res, open("runs/v_timecourse.json", "w"))
