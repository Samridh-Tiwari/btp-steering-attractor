"""Why are LFADS's input directions blind to its memory? In a GRU a unit holds its value
when its update gate z is near 1, and the same gate scales the input's effect by (1 - z).
Check: during the delay, where does the behavioural sensitivity s sit (by z), and where do
the controller's push directions (dg/dco) sit?"""
import sys, json
import numpy as np, torch
import lfads_full as L, lfads_tests as LT, steer as S

def zgate(cell, u, h):
    x_z = (u @ cell.weight_ih.T + cell.bias_ih)[:, :cell.hidden_size]
    h_z = (h @ cell.weight_hh.T + cell.bias_hh)[:, :cell.hidden_size]
    return torch.sigmoid(x_z + h_z)

def check(path, data, rows=None):
    m, ck = LT.load(path)
    if rows is not None:
        data = L.subset(data, np.asarray(rows))
    tl, tg = data["t_load"], data["t_go"]; n = len(tg)
    o = L.infer(m, data); G, U, g0 = o["gen"], o["co_mean"], o["g0"]
    w, b, _ = LT.lever(G, data["z"])
    s = LT.sensitivity(m, g0, U, w, b, tg)
    cell = m.gen_cell(); idx = S.window_idx("delay", tl, tg)
    s_hi, j_hi, frac_hi, zs, zw_s, zw_j = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, n, 2):
            for t in idx[i][::5]:
                z = zgate(cell, U[i, t][None], G[i, t - 1][None])[0].numpy()
                Ju = LT.cell_jac_u(cell, U[i, t], G[i, t - 1])
                hi = z > 0.95
                st = s[i, t] ** 2; jn = (Ju ** 2).sum(1)
                s_hi.append(st[hi].sum() / st.sum()); j_hi.append(jn[hi].sum() / jn.sum()); frac_hi.append(hi.mean())
                zw_s.append((z * st).sum() / st.sum()); zw_j.append((z * jn).sum() / jn.sum())
    return dict(frac_units_z95=float(np.median(frac_hi)), sens_share_on_z95=float(np.median(s_hi)),
                push_share_on_z95=float(np.median(j_hi)), z_weighted_by_sens=float(np.median(zw_s)),
                z_weighted_by_push=float(np.median(zw_j)))

if __name__ == "__main__":
    torch.set_num_threads(1)
    out = {}
    for p in sys.argv[1:]:
        ck = torch.load(p, weights_only=False); data = torch.load(ck["data"], weights_only=False)
        rows = ck["eval"].get("test_idx")
        out[p] = check(p, data, rows); print(p, {k: round(v, 3) for k, v in out[p].items()}, flush=True)
    try:
        old = json.load(open("runs/gate_check.json")); old.update(out); out = old
    except Exception:
        pass
    json.dump(out, open("runs/gate_check.json", "w"), indent=1)
