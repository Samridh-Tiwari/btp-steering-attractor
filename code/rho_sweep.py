"""
What does it take to get 0.1%?  Error removed against the effort cost rho,
for the behaviour goal and the whole-state goal, on the teacher.

rho is expressed relative to rho0, the value at which driving every unit
removes 90% (the normalisation used everywhere else in the report).
"""
import json
import numpy as np
import torch

import control as C
import steer as S
import state_target as ST
from teacher import load_teacher, cued_response

torch.set_num_threads(1)
net, _ = load_teacher("runs/teacher_g0.9_n0.05_s0.pt")
n = 48
rng = np.random.default_rng(21)
u, z, m, meta = cued_response(n, rng=rng, return_meta=True)
tg = meta["t_go"]
s, _ = S.teacher_sensitivity(net, u, tg)
idx = [list(range(t - 30, t - 5)) for t in tg]
t_end = np.array([t - 5 for t in tg])


def step_fn(x):
    h = torch.zeros(n, net.n_hid); H = []
    for t in range(u.shape[1]):
        h = torch.tanh(h @ net.W_rec.T + u[:, t] @ net.W_in.T + net.b + x[:, t]); H.append(h)
    return torch.stack(H, 1)


Ms = ST.end_state_jacobians(step_fn, n, u.shape[1], net.n_hid, idx, t_end)
Qs = S.Q_from(s, idx)
mult = np.logspace(-2, 5, 29)
Win = net.W_in.detach().numpy()
res = {"mult": mult.tolist()}
r2 = np.random.default_rng(0)
for name in ("real_inputs", "random_4", "best_4_behaviour"):
    beh, sta = [], []
    for i in range(n):
        N = Ms.shape[-1]
        if name == "real_inputs":
            P = C.orth(Win)
        elif name == "random_4":
            P = np.linalg.qr(r2.standard_normal((N, 4)))[0]
        else:
            P = np.linalg.eigh((Qs[i] + Qs[i].T) / 2)[1][:, -4:]
        W_I = np.einsum("tij,tkj->ik", Ms[i], Ms[i])
        rho_s0 = ST.solve_rho(np.linalg.eigvalsh((W_I + W_I.T) / 2))
        rho_b0 = np.trace(Qs[i]) / 9.0
        beh.append([ST.er_behaviour(Qs[i], P, rho_b0 * k) for k in mult])
        sta.append([ST.er_state(Ms[i], P, rho_s0 * k) for k in mult])
    res[name] = dict(behaviour=np.median(beh, 0).tolist(), state=np.median(sta, 0).tolist())
    b, st_ = np.median(beh, 0), np.median(sta, 0)
    def cross(y):
        j = np.where(y < 1e-3)[0]
        return float(mult[j[0]]) if len(j) else None
    res[name]["mult_to_0.1pct"] = dict(behaviour=cross(b), state=cross(st_))
    print(name, "at rho0: behaviour %.1f%%  state %.1f%%" % (b[np.argmin(abs(mult - 1))] * 100,
                                                             st_[np.argmin(abs(mult - 1))] * 100),
          "| rho multiple for 0.1%:", res[name]["mult_to_0.1pct"])
json.dump(res, open("runs/rho_sweep.json", "w"), indent=1)
