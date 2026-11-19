"""
The TEACHER network: a stand-in for mouse motor cortex with known ground truth.

TASK (cued response, built to mirror the lever-pull experiment)
---------------------------------------------------------------
Two input channels, one output, T = 150 steps (like the 150 bins of the
SURF trials).

  channel 0, "value": a 5-step pulse of amplitude v, v ~ U[0.2, 1.0], at a
             random time in [10, 20). This loads the PLAN — how hard to pull.
  channel 1, "go":    a 3-step unit pulse at a random time in [60, 100).
             This is the CUE.
  output, "lever":    0 until the go cue, then v * pullshape(t - t_go):
             rise over 8 steps, plateau 20, fall over 8, then back to 0.

To solve this the network must (1) hold an arbitrary real number v through a
delay of 35-85 steps with no input — which forces a line attractor — and
(2) on the go cue, generate a movement whose size is set by where it sits on
that attractor. Same structure as the real task: a plan held on a manifold,
a cue that triggers execution, behaviour that depends on the held plan.

TRAINING
--------
Vanilla tanh RNN (same as model.py). Two departures from the earlier
integrator runs, both because of what those runs taught us:
  * noise is injected into the hidden state during training. Plain backprop
    at g = 1.0 produced a chaotic memory with no fixed points; noise is the
    standard way to push training toward robust attractors instead.
  * the initial spectral scale g is a parameter, swept, and the teacher kept
    is the one whose delay-period dynamics settle onto genuine fixed points.

Usage:  python teacher.py --g 0.9 --noise 0.05 --seed 0
"""

import argparse
import json
import os
import numpy as np
import torch

from model import VanillaRNN

T_TRIAL = 150


def pullshape(n):
    """Rise 8, plateau 20, fall 8. Returns array length n (zero-padded)."""
    s = np.concatenate([np.linspace(0, 1, 9)[1:], np.ones(20), np.linspace(1, 0, 9)[1:]])
    out = np.zeros(n)
    out[:min(n, len(s))] = s[:n]
    return out


def cued_response(batch=64, T=T_TRIAL, rng=None, return_meta=False, hint=False):
    """
    Returns u (B,T,2), z (B,T,1 or 2), m as float32 torch tensors.

    hint=True adds a second output channel used ONLY during teacher training:
    it reports v from the end of the load pulse until the go cue. Without it,
    training stalls at MSE ~0.0088 — exactly the error of a network that
    pulls at the right time with average strength and ignores v — because the
    plan is only needed 35-85 steps after it is loaded and the gradient has to
    cross that whole delay. The hint gives storing v immediate credit. The
    student never sees it and the analysis never uses this readout.
    """
    rng = np.random.default_rng() if rng is None else rng
    u = np.zeros((batch, T, 2), np.float32)
    z = np.zeros((batch, T, 2 if hint else 1), np.float32)
    v = rng.uniform(0.2, 1.0, batch)
    t_load = rng.integers(10, 20, batch)
    t_go = rng.integers(60, 100, batch)
    for b in range(batch):
        u[b, t_load[b]:t_load[b] + 5, 0] = v[b]
        u[b, t_go[b]:t_go[b] + 3, 1] = 1.0
        z[b, t_go[b]:, 0] = v[b] * pullshape(T - t_go[b])
        if hint:
            z[b, t_load[b] + 5:t_go[b], 1] = v[b]
    m = np.ones_like(z)
    out = (torch.from_numpy(u), torch.from_numpy(z), torch.from_numpy(m))
    if return_meta:
        return out + (dict(v=v, t_load=t_load, t_go=t_go),)
    return out


def run_noisy(net, u, noise=0.0, h0=None):
    """Forward pass with optional hidden-state noise (training only)."""
    B, T, _ = u.shape
    h = torch.zeros(B, net.n_hid) if h0 is None else h0
    H = []
    for t in range(T):
        pre = h @ net.W_rec.T + u[:, t] @ net.W_in.T + net.b
        if noise > 0:
            pre = pre + noise * torch.randn_like(pre)
        h = torch.tanh(pre)
        H.append(h)
    H = torch.stack(H, 1)
    return net.readout(H), H


def train_teacher(g=0.9, noise=0.05, n_hid=100, steps=6000, lr=1e-3, reg_h=1e-4,
                  seed=0, outdir="runs", log_every=500):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = VanillaRNN(2, n_hid, 2, g=g, input_scale=1.0, seed=seed)   # out 0 lever, 1 hint
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    for it in range(1, steps + 1):
        u, z, m = cued_response(64, rng=rng, hint=True)
        y, H = run_noisy(net, u, noise)
        mse_lever = ((y[..., 0] - z[..., 0]) ** 2).mean()
        mse_hint = ((y[..., 1] - z[..., 1]) ** 2).mean()
        loss = mse_lever + 0.5 * mse_hint + reg_h * (H ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if it % log_every == 0 or it == 1:
            print(f"  step {it:5d}  lever mse {mse_lever.item():.5f}  hint mse {mse_hint.item():.5f}",
                  flush=True)
    os.makedirs(outdir, exist_ok=True)
    tag = f"teacher_g{g}_n{noise}_s{seed}"
    torch.save({"state_dict": net.state_dict(),
                "cfg": dict(g=g, noise=noise, n_hid=n_hid, steps=steps, seed=seed)},
               os.path.join(outdir, tag + ".pt"))
    return net, tag


def load_teacher(path):
    ck = torch.load(path, weights_only=False)
    c = ck["cfg"]
    net = VanillaRNN(2, c["n_hid"], 2, g=c["g"])
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, c


@torch.no_grad()
def diagnose(net, n=512, seed=123, settle=3000):
    """Everything we need to know before trusting this as ground truth."""
    rng = np.random.default_rng(seed)
    u, z, m, meta = cued_response(n, rng=rng, return_meta=True)
    y, H = run_noisy(net, u, 0.0)
    y = y[..., :1]                                  # lever only; hint is discarded
    v, t_go = meta["v"], meta["t_go"]

    # task performance
    mse = ((y - z) ** 2).mean().item()
    r2_out = 1 - ((y - z) ** 2).sum().item() / ((z - z.mean()) ** 2).sum().item()
    # peak pull amplitude vs v (does behaviour follow the plan?)
    peak = np.array([y[b, t_go[b] + 8:t_go[b] + 28, 0].mean().item() for b in range(n)])
    r_peak_v = np.corrcoef(peak, v)[0, 1]

    # plan held through the delay: decode v from state just before go
    Hgo = torch.stack([H[b, t_go[b] - 1] for b in range(n)]).numpy()
    X = np.c_[Hgo, np.ones(n)]
    wv = np.linalg.lstsq(X, v, rcond=None)[0]
    r2_plan = 1 - ((X @ wv - v) ** 2).sum() / ((v - v.mean()) ** 2).sum()

    # do delay states settle onto fixed points? relax autonomously from mid-delay
    tmid = np.minimum(t_go - 10, 55)
    h = torch.stack([H[b, tmid[b]] for b in range(n)])
    for _ in range(settle):
        h = net.f_autonomous(h)
    speed = (net.f_autonomous(h) - h).norm(dim=1)
    Hs = h.numpy()
    Hc = Hs - Hs.mean(0)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    pr = (S ** 2).sum() ** 2 / (S ** 4).sum()
    # does position along the settled line still encode v?
    pc1 = Hc @ Vt[0]
    r_pc1_v = abs(np.corrcoef(pc1, v)[0, 1])

    # slow eigenvalues at settled points and alignment with the line
    mags, angs = [], []
    for i in range(0, n, max(1, n // 40)):
        J = net.jacobian(torch.from_numpy(Hs[i]))
        ev, evec = np.linalg.eig(J.numpy())
        o = np.argsort(-np.abs(ev))
        mags.append(np.abs(ev[o[:4]]))
        e1 = np.real(evec[:, o[0]]); e1 /= np.linalg.norm(e1)
        angs.append(np.degrees(np.arccos(min(1.0, abs(e1 @ Vt[0])))))
    mags = np.array(mags)

    return dict(mse=mse, r2_out=r2_out, r_peak_v=r_peak_v, r2_plan=r2_plan,
                speed_median=float(speed.median()), speed_max=float(speed.max()),
                pc_var=var[:5].tolist(), pr=float(pr), r_pc1_v=float(r_pc1_v),
                lam_top=mags.mean(0).tolist(), lam1_sd=float(mags[:, 0].std()),
                angle_slow_vs_line=float(np.median(angs)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--g", type=float, default=0.9)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(1)
    print(f"teacher g={a.g} noise={a.noise} seed={a.seed}", flush=True)
    net, tag = train_teacher(g=a.g, noise=a.noise, steps=a.steps, seed=a.seed)
    d = diagnose(net)
    print(json.dumps(d, indent=2))
    with open(f"runs/{tag}_diag.json", "w") as f:
        json.dump(d, f, indent=2)
