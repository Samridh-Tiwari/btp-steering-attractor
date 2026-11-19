"""
A student that is TOLD when inputs can arrive ("event student").

Same generator, factors and Poisson readout as lfads_lite.Student. The only change is
the input model. The free student has a free input u_i(t) on every trial at every time
step. Here the model is given the two event times of each trial (as an experimenter
would know when the cue and the go tone were played) and may only inject input locked
to them:

    g(t+1) = tanh( W g(t) + b + sum_c a_ic * K_load_c[t - t_load_i] + K_go[t - t_go_i] )

  K_go    (L_go, N)  one go-locked waveform shared by ALL trials: it can trigger the pull
                     but cannot carry anything trial-specific.
  K_load_c (L_load, N), c = 1..n_amp, shared waveforms scaled by per-trial amplitudes a_ic:
                     the only trial-specific input, and it ends L_load steps after the cue.

So whatever distinguishes trials at the time of the pull has to be carried through the
delay by the generator's own dynamics. This is the opposite prior to the free student.
It is the right prior for teacher A (pulse, then silence) and the wrong one for teacher B
(input held on through the delay).

    python event_student.py --data runs/dataset.pt --seed 0 [--suffix _B]
"""
import argparse
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn

from lfads_lite import poisson_nll, bits_per_spike, fit_glm

L_GO = (-3, 12)       # lags relative to go at which the shared go waveform may be non-zero
L_LOAD = (-1, 7)      # lags relative to the load cue


class EventStudent(nn.Module):
    def __init__(self, n_trials, T, n_obs, t_load, t_go, N=64, n_fac=12, n_amp=2, g=0.9, seed=0, mean_rate=None):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.N, self.T, self.n_amp = N, T, n_amp
        self.W = nn.Parameter(torch.randn(N, N, generator=gen) * g / math.sqrt(N))
        self.b = nn.Parameter(torch.zeros(N))
        self.F = nn.Parameter(torch.randn(n_fac, N, generator=gen) / math.sqrt(N))
        self.Dout = nn.Parameter(torch.randn(n_obs, n_fac, generator=gen) * 0.3)
        d0 = torch.log(mean_rate) if mean_rate is not None else torch.zeros(n_obs)
        self.d = nn.Parameter(d0.clone())
        self.g0 = nn.Parameter(torch.randn(1, N, generator=gen) * 0.1)          # ONE start state, shared
        self.Kgo = nn.Parameter(torch.randn(L_GO[1] - L_GO[0], N, generator=gen) * 0.1)
        self.Kload = nn.Parameter(torch.randn(n_amp, L_LOAD[1] - L_LOAD[0], N, generator=gen) * 0.1)
        self.amp = nn.Parameter(torch.randn(n_trials, n_amp, generator=gen) * 0.1)
        # index tensors: for each trial and time, which lag of each kernel applies (-1 = none)
        ig = torch.full((n_trials, T), -1, dtype=torch.long); il = ig.clone()
        for i in range(n_trials):
            for t in range(T):
                lg = t - int(t_go[i]) - L_GO[0]
                if 0 <= lg < L_GO[1] - L_GO[0]:
                    ig[i, t] = lg
                ll = t - int(t_load[i]) - L_LOAD[0]
                if 0 <= ll < L_LOAD[1] - L_LOAD[0]:
                    il[i, t] = ll
        Lg = L_GO[1] - L_GO[0]; Ll = L_LOAD[1] - L_LOAD[0]
        og = torch.zeros(n_trials, T, Lg); ol = torch.zeros(n_trials, T, Ll)
        for i in range(n_trials):
            for t in range(T):
                if ig[i, t] >= 0: og[i, t, ig[i, t]] = 1.0
                if il[i, t] >= 0: ol[i, t, il[i, t]] = 1.0
        self.register_buffer("og", og); self.register_buffer("ol", ol)

    def drive(self, amp=None, rows=None):
        """Total input to the pre-activation, (n, T, N)."""
        amp = self.amp if amp is None else amp
        og = self.og if rows is None else self.og[rows]; ol = self.ol if rows is None else self.ol[rows]
        dg = og @ self.Kgo                                                      # (n,T,N)
        dl = sum(amp[:, c, None, None] * (ol @ self.Kload[c]) for c in range(self.n_amp))
        return dg + dl

    def generate(self, use_u=True, drive=None, g0=None, t0=0):
        n = self.amp.shape[0] if drive is None else drive.shape[0]
        drive = self.drive() if drive is None else drive
        g = torch.tanh(self.g0).expand(n, -1) if g0 is None else g0
        dr = drive.unbind(1)
        G, P = [], []
        for t in range(self.T):
            if t < t0:
                G.append(torch.zeros(n, self.N)); P.append(torch.zeros(n, self.N)); continue
            pre = g @ self.W.T + self.b
            if use_u:
                pre = pre + dr[t]
            g = torch.tanh(pre)
            G.append(g); P.append(pre)
        return torch.stack(G, 1), torch.stack(P, 1)

    def logrates(self, G):
        return (G @ self.F.T) @ self.Dout.T + self.d


def fit(data, N=64, steps=2500, lr=1e-2, seed=0, lam_w=1e-5, lam_k=1e-3, log_every=250):
    torch.manual_seed(seed)
    S = data["spikes"]; nin = data["n_heldin"]; Sin = S[..., :nin]
    n, T, _ = S.shape
    st = EventStudent(n, T, nin, data["t_load"], data["t_go"], N=N, seed=seed,
                      mean_rate=Sin.mean(dim=(0, 1)).clamp(min=1e-3))
    opt = torch.optim.Adam(st.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    t0 = time.time()
    for it in range(1, steps + 1):
        G, _ = st.generate()
        lr_ = st.logrates(G)
        nll = poisson_nll(lr_, Sin)
        loss = nll + lam_w * (st.W ** 2).sum() + lam_k * ((st.Kgo ** 2).mean() + (st.Kload ** 2).mean() + (st.amp ** 2).mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(st.parameters(), 5.0)
        opt.step(); sched.step()
        if it % log_every == 0 or it == 1:
            print(f"  step {it:5d}  nll {nll.item():.4f}  bps {bits_per_spike(lr_.detach(), Sin):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return st


@torch.no_grad()
def evaluate(st, data):
    S = data["spikes"]; nin = data["n_heldin"]
    G, _ = st.generate()
    out = dict(bps_heldin=bits_per_spike(st.logrates(G), S[..., :nin]))
    fac = (G @ st.F.T).detach()
    with torch.enable_grad():
        lr_out = fit_glm(fac, S[..., nin:])
    out["bps_cosmooth"] = bits_per_spike(lr_out, S[..., nin:])
    return out


def load(path, data):
    ck = torch.load(path, weights_only=False)
    n, T, _ = data["spikes"].shape
    st = EventStudent(n, T, data["n_heldin"], data["t_load"], data["t_go"], N=ck["N"])
    st.load_state_dict(ck["state_dict"]); st.eval()
    return st, ck


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/dataset.pt")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suffix", default="")
    a = p.parse_args()
    torch.set_num_threads(1)
    data = torch.load(a.data, weights_only=False)
    tag = f"event_s{a.seed}{a.suffix}"
    print(tag, flush=True)
    st = fit(data, steps=a.steps, seed=a.seed)
    ev = evaluate(st, data)
    print(json.dumps(ev, indent=1))
    torch.save({"state_dict": st.state_dict(), "seed": a.seed, "N": st.N, "eval": ev, "kind": "event"}, f"runs/{tag}.pt")
