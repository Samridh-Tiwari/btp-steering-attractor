"""
Full-LFADS fits to MC_Maze: steering authority of the inferred channels for the reach end
point, plus the swap / zero / persistence tests, as mc_maze.py and mc_plan_tests.py did
for the simplified student. Pushes are added to the generator state (as in lfads_tests).

    python lfads_mc_tests.py runs/lfads_MC_*.pt
"""
import json
import sys

import numpy as np
import torch

import lfads_full as L
import lfads_tests as LT
import mc_maze as M
import control as C
from plan_tests import persistence


def reach_feats(G, tm):
    return np.stack([G[i, t - M.LAG: t - M.LAG + M.REACH_BINS].mean(0) for i, t in enumerate(tm)])


def reach_decoder(G, data, ridge=1e-3, folds=5):
    tm = data["t_move"]; R = data["reach"]
    Fm = reach_feats(G, tm)
    def solve(F, Y):
        mu, ym = F.mean(0), Y.mean(0)
        W = np.linalg.solve((F - mu).T @ (F - mu) + ridge * len(F) * np.eye(F.shape[1]), (F - mu).T @ (Y - ym))
        return W, ym - mu @ W
    idx = np.random.default_rng(0).permutation(len(Fm)); P = np.zeros_like(R)
    for f in np.array_split(idx, folds):
        m = np.ones(len(Fm), bool); m[f] = False
        W, b = solve(Fm[m], R[m]); P[f] = Fm[f] @ W + b
    r2 = 1 - ((P - R) ** 2).sum(0) / ((R - R.mean(0)) ** 2).sum(0)
    W, b = solve(Fm, R)
    return W, b, r2


def sens(m, g0, U, W, b, tm):
    cell = m.gen_cell(); n, T, _ = U.shape
    mask = torch.zeros(n, T)
    for i, t in enumerate(tm):
        mask[i, t - M.LAG: t - M.LAG + M.REACH_BINS] = 1.0 / M.REACH_BINS
    Wt = torch.tensor(W, dtype=torch.float32); bt = torch.tensor(b, dtype=torch.float32)
    out = []
    for c in range(2):
        x = torch.zeros(n, T, m.hps.gen_dim, requires_grad=True)
        g = g0.detach(); ys = []
        for t in range(T):
            g = cell(U[:, t], g) + x[:, t]
            ys.append(g @ Wt[:, c] + bt[c])
        ((torch.stack(ys, 1) * mask).sum(1)).sum().backward()
        out.append(x.grad.detach().numpy())
    return out


def tests(path, data):
    m, ck = LT.load(path)
    n, T, _ = data["spikes"].shape; nb = data["pre_go_bins"]; tm = data["t_move"]
    o = L.infer(m, data)
    G, U, g0 = o["gen"], o["co_mean"], o["g0"]
    W, b, r2 = reach_decoder(G.numpy(), data)
    res = dict(path=path, kl_co=ck["kl_co"], seed=ck["seed"], heldin=ck["eval"]["bps_heldin"],
               cosmooth=ck["eval"]["bps_cosmooth"], autonomy=ck["eval"]["autonomy"], reach_r2=r2.tolist())
    perm = np.random.default_rng(1).permutation(n)
    R0 = reach_feats(G.numpy(), tm) @ W + b
    def reach(UU):
        return reach_feats(m.run_generator(g0, UU).numpy(), tm) @ W + b
    def slopes(R):
        out = {}
        for c, nm in enumerate("xy"):
            X = np.c_[R0[:, c], R0[perm, c], np.ones(n)]
            w = np.linalg.lstsq(X, R[:, c], rcond=None)[0]
            out[nm] = [float(w[0]), float(w[1])]
        return out
    Ug = U.clone(); Ug[:, nb - 2:] = U[perm][:, nb - 2:]
    res["swap_go"] = slopes(reach(Ug)); res["swap_all"] = slopes(reach(U[perm]))
    Rz = reach(torch.zeros_like(U))
    res["zero_all"] = {nm: float(np.polyfit(R0[:, c], Rz[:, c], 1)[0]) for c, nm in enumerate("xy")}
    cell = m.gen_cell(); mags = []
    for i in range(0, n, 12):
        Js = [LT.cell_jac_h(cell, U[i, t], G[i, t - 1]) for t in range(1, nb - 1)]
        mg, _ = persistence(Js); mags.append(mg[:4])
    res["persist_top"] = np.median(mags, 0).tolist()
    pw = (U.numpy() ** 2).sum(-1); res["u_share_go"] = float(pw[:, nb - 2:].sum() / pw.sum())
    # authority, last 300 ms of the delay
    sx = sens(m, g0, U, W, b, tm)
    idx = list(range(1, nb - 2)); k = m.hps.co_dim
    rng = np.random.default_rng(0); rows = []
    for i in range(0, n, 2):
        Q = sum(s[i, idx].T @ s[i, idx] for s in sx)
        Gf = float(np.trace(Q))
        if Gf <= 0:
            continue
        Ju = [LT.cell_jac_u(cell, U[i, t], G[i, t - 1]) for t in idx]
        P = np.linalg.svd(np.concatenate(Ju, 1), full_matrices=False)[0][:, :k]
        g = C.G_of(Q, P); gr = C.G_random(Q, k, 200, rng)
        ev = np.linalg.eigvalsh((Q + Q.T) / 2)[::-1]
        rows.append(dict(frac_full=g / Gf, frac_rand=float(gr.mean()) / Gf, frac_opt=float(ev[:k].sum()) / Gf,
                         frac_opt1=float(ev[0]) / Gf, err_red=C.error_reduction(g, Gf / 9)))
    res["authority"] = {key: float(np.median([r_[key] for r_ in rows])) for key in rows[0]}
    return res


if __name__ == "__main__":
    torch.set_num_threads(1)
    data = torch.load("runs/mc_data.pt", weights_only=False)
    out_path = "runs/lfads_mc_tests.json"
    try:
        allres = json.load(open(out_path))
    except Exception:
        allres = {}
    for p in sys.argv[1:]:
        r = tests(p, data); allres[p] = r
        json.dump(allres, open(out_path, "w"), indent=1)
        a = r["authority"]
        print(f"{p}: cs {r['cosmooth']:.3f} auton {r['autonomy']:.2f} reach R2 {np.round(r['reach_r2'], 2)} | "
              f"auth {a['frac_full']*100:.2f}% rand {a['frac_rand']*100:.1f}% best4 {a['frac_opt']*100:.0f}% best1 {a['frac_opt1']*100:.0f}% "
              f"err {a['err_red']*100:.2f}% | swap_go x {np.round(r['swap_go']['x'], 2)} all x {np.round(r['swap_all']['x'], 2)} "
              f"zero {r['zero_all']} | persist {np.round(r['persist_top'][:2], 2)} u_go {r['u_share_go']:.2f}", flush=True)
