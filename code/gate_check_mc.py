"""Gate check on the full-LFADS MC_Maze fits (delay window, reach sensitivity)."""
import json, sys
import numpy as np, torch
import lfads_full as L, lfads_tests as LT, lfads_mc_tests as MT
from gate_check import zgate
torch.set_num_threads(1)
data = torch.load("runs/mc_data.pt", weights_only=False)
nb = data["pre_go_bins"]; out = {}
for p in sys.argv[1:]:
    m, ck = LT.load(p); o = L.infer(m, data); G, U, g0 = o["gen"], o["co_mean"], o["g0"]
    W, b, _ = MT.reach_decoder(G.numpy(), data)
    sx = MT.sens(m, g0, U, W, b, data["t_move"])
    cell = m.gen_cell(); n = G.shape[0]; rows = []
    with torch.no_grad():
        for i in range(0, n, 3):
            for t in range(1, nb - 2, 3):
                z = zgate(cell, U[i, t][None], G[i, t - 1][None])[0].numpy()
                Ju = LT.cell_jac_u(cell, U[i, t], G[i, t - 1])
                hi = z > 0.95; st = sum(s[i, t] ** 2 for s in sx); jn = (Ju ** 2).sum(1)
                rows.append([hi.mean(), st[hi].sum() / st.sum(), jn[hi].sum() / jn.sum()])
    md = np.median(rows, 0)
    out[p] = dict(frac_units_z95=float(md[0]), sens_share_on_z95=float(md[1]), push_share_on_z95=float(md[2]))
    print(p, {k: round(v, 3) for k, v in out[p].items()}, flush=True)
json.dump(out, open("runs/gate_check_mc.json", "w"), indent=1)
