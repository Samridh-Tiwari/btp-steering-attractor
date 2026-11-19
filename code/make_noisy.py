"""Teacher A recordings with noise in the circuit's own dynamics (sigma = 0.1 per step on
every unit's input). Same trials as dataset.pt (task seed 7). The noise realisation is
stored, so the true push sensitivity can be computed along the actual noisy path."""
import numpy as np, torch
from teacher import load_teacher, cued_response
SIG = 0.1
net, _ = load_teacher("runs/teacher_g0.9_n0.05_s0.pt")
rng = np.random.default_rng(7)
u, z, m, meta = cued_response(256, rng=rng, return_meta=True)
n, T, _ = u.shape
torch.manual_seed(11)
xi = SIG * torch.randn(n, T, net.n_hid)
h = torch.zeros(n, net.n_hid); Hs = []
with torch.no_grad():
    for t in range(T):
        h = torch.tanh(h @ net.W_rec.T + u[:, t] @ net.W_in.T + net.b + xi[:, t]); Hs.append(h)
H = torch.stack(Hs, 1).numpy()
n_obs, n_heldout, mod_sd = 60, 10, 0.7
C = rng.standard_normal((n_obs, H.shape[-1]))
X = H.reshape(-1, H.shape[-1]) @ C.T
X = (X - X.mean(0)) / X.std(0)
base = rng.uniform(0.08, 0.45, n_obs)
logr = np.log(base) + mod_sd * X - mod_sd ** 2 / 2
rates = np.exp(logr).reshape(n, -1, n_obs)
spikes = rng.poisson(rates).astype(np.float32)
with torch.no_grad():
    y = net.readout(torch.from_numpy(H))
torch.save(dict(spikes=torch.from_numpy(spikes), rates=rates.astype(np.float32), H_teacher=H.astype(np.float32),
                u_teacher=u.numpy(), z=z.numpy(), v=meta["v"], t_go=meta["t_go"], t_load=meta["t_load"],
                n_heldin=n_obs - n_heldout, C=C, xi=xi.numpy(), sigma=SIG, z_actual=y.numpy()), "runs/dataset_An.pt")
print("saved", spikes.mean())
