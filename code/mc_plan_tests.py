"""Plan-location tests on the MC_Maze students (no ground truth here).
swap: give trial i the controller inputs of trial j, either for the whole window ("all")
or from 40 ms before the go cue onwards ("go"). Does the decoded reach follow trial i's own
start state / delay state, or the partner's inputs?  zero: controller off for the whole window.
persistence: eigenvalues of the propagator across the 14 delay bins."""
import glob, json, sys
import numpy as np, torch
from lfads_lite import Student
import mc_maze as M
from plan_tests import persistence

def main():
    torch.set_num_threads(1)
    data = torch.load("runs/mc_data.pt", weights_only=False)
    n, T, _ = data["spikes"].shape; nb = data["pre_go_bins"]; tm = data["t_move"]
    perm = np.random.default_rng(1).permutation(n)
    out = []
    for p in sorted(glob.glob("runs/mc_student_*_s?.pt")):
        ck = torch.load(p, weights_only=False)
        st = Student(n, T, data["n_heldin"], N=ck["N"], k=ck["k"]); st.load_state_dict(ck["state_dict"]); st.eval()
        W, b, r2 = M.reach_decoder(st, data, 1e-3)
        def reach(U):
            with torch.no_grad():
                G, P = st.generate(use_u=True, U=U)
            return np.stack([(G[i, t - M.LAG: t - M.LAG + M.REACH_BINS].mean(0) @ W + b).numpy() for i, t in enumerate(tm)]), P
        U = st.U.detach(); R0, P = reach(U)
        res = dict(path=p, k=ck["k"], lam=ck["lam_u"], seed=ck["seed"], cosmooth=ck["eval"]["bps_cosmooth"], reach_r2=r2.tolist())
        def slopes(R):
            o = {}
            for c, nm in enumerate("xy"):
                X = np.c_[R0[:, c], R0[perm, c], np.ones(n)]
                w = np.linalg.lstsq(X, R[:, c], rcond=None)[0]
                o[nm] = [float(w[0]), float(w[1])]
            return o
        Ug = U.clone(); Ug[:, nb - 2:] = U[perm][:, nb - 2:]
        res["swap_go"] = slopes(reach(Ug)[0])
        res["swap_all"] = slopes(reach(U[perm])[0])
        Rz = reach(torch.zeros_like(U))[0]
        res["zero_all"] = {nm: float(np.polyfit(R0[:, c], Rz[:, c], 1)[0]) for c, nm in enumerate("xy")}
        Wn = st.W.detach().numpy(); Pn = P.numpy(); mags = []; cnt = []
        for i in range(0, n, 12):
            m, _ = persistence([(1 - np.tanh(Pn[i, t]) ** 2)[:, None] * Wn for t in range(1, nb - 1)])
            mags.append(m[:4])
            e1 = np.abs(np.linalg.eigvals((1 - np.tanh(Pn[i, nb // 2]) ** 2)[:, None] * Wn)); cnt.append((e1 > 0.9).sum())
        res["persist_top"] = np.median(mags, 0).tolist(); res["n_persist_05"] = float(np.median([(m > 0.5).sum() for m in mags]))
        res["n_slow_09"] = float(np.median(cnt))
        pw = (U.numpy() ** 2).sum(-1); res["u_share_go"] = float(pw[:, nb - 2:].sum() / pw.sum())
        out.append(res)
        print(f"k{res['k']} lam {res['lam']:g} s{res['seed']} cs {res['cosmooth']:.3f}: swap_go x own/partner {res['swap_go']['x'][0]:.2f}/{res['swap_go']['x'][1]:.2f}"
              f" y {res['swap_go']['y'][0]:.2f}/{res['swap_go']['y'][1]:.2f} | swap_all x {res['swap_all']['x'][0]:.2f}/{res['swap_all']['x'][1]:.2f}"
              f" y {res['swap_all']['y'][0]:.2f}/{res['swap_all']['y'][1]:.2f} | zero x {res['zero_all']['x']:.2f} y {res['zero_all']['y']:.2f}"
              f" | persist {np.round(res['persist_top'][:2],2)} n>.5 {res['n_persist_05']:g} n|lam|>.9 {res['n_slow_09']:g}", flush=True)
    json.dump(out, open("runs/mc_plan_tests.json", "w"), indent=1)
main()
