"""
Full LFADS student, built from the lfads-torch modules (Sedler and Pandarinath 2023),
with a plain training loop instead of Lightning/Hydra.

Same pieces as the LFADS used in the SURF project:
  IC encoder      bidirectional ClippedGRU over the whole trial -> posterior over g0
  CI encoder      bidirectional ClippedGRU -> controller inputs (lag 1)
  controller      ClippedGRU fed (CI, previous factors) -> posterior over co(t)
  generator       ClippedGRU driven by co(t)
  factors         KernelNormalizedLinear (unit-norm rows), readout FanInLinear, Poisson
  priors          IC: Gaussian (learned mean, var 0.1); CO: AR(1) (tau 10, nvar 0.1, learned)
  loss            recon (mean) + L2 ramp + KL ramp, scaled by 1e4; coordinated dropout 0.3

Only the sizes are smaller (CPU, 50 neurons).

    python lfads_full.py --data runs/dataset.pt --kl_co 1e-4 --seed 0 --tag A
"""
import argparse
import os
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.environ.get("LFADS_TORCH_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lfads-torch")))
from lfads_torch.modules.encoder import Encoder                       # noqa: E402
from lfads_torch.modules.decoder import Decoder                       # noqa: E402
from lfads_torch.modules.priors import (MultivariateNormal,           # noqa: E402
                                        AutoregressiveMultivariateNormal)
from lfads_torch.modules.readin_readout import FanInLinear            # noqa: E402
from lfads_torch.modules.augmentations import CoordinatedDropout      # noqa: E402

from lfads_lite import bits_per_spike, fit_glm                        # noqa: E402


def make_hps(n_in, T, co_dim=4, gen_dim=64, con_dim=32, fac_dim=12, ic_dim=32,
             ic_enc_dim=32, ci_enc_dim=32, dropout=0.1, ic_win=0):
    return SimpleNamespace(
        encod_data_dim=n_in, encod_seq_len=T, recon_seq_len=T, ext_input_dim=0,
        ic_enc_seq_len=ic_win, ic_enc_dim=ic_enc_dim, ci_enc_dim=ci_enc_dim, ci_lag=1,
        con_dim=con_dim, co_dim=co_dim, ic_dim=ic_dim, gen_dim=gen_dim, fac_dim=fac_dim,
        dropout_rate=dropout, cell_clip=5.0, ic_post_var_min=1e-4)


class LFADS(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.co_prior = AutoregressiveMultivariateNormal(tau=10.0, nvar=0.1, shape=hps.co_dim)
        self.ic_prior = MultivariateNormal(mean=0.0, variance=0.1, shape=hps.ic_dim)
        hps.co_prior = self.co_prior          # DecoderCell reads the posterior family from here
        self.encoder = Encoder(hps)
        self.decoder = Decoder(hps)
        self.readout = FanInLinear(hps.fac_dim, hps.encod_data_dim)

    def forward(self, x, sample=False, sample_co=None):
        sample_co = sample if sample_co is None else sample_co
        n = x.shape[0]
        ic_mean, ic_std, ci = self.encoder(x)
        ic = self.ic_prior.make_posterior(ic_mean, ic_std).rsample() if sample else ic_mean
        ext = torch.zeros(n, x.shape[1], 0)
        g0, gen, con, co_m, co_s, gin, fac = self.decoder(ic, ci, ext, sample_posteriors=sample_co)
        logr = self.readout(fac)
        return dict(ic_mean=ic_mean, ic_std=ic_std, ci=ci, g0=g0, gen=gen, con=con,
                    co_mean=co_m, co_std=co_s, gen_input=gin, factors=fac, logr=logr)

    # ---------------------------------------------------------- open-loop generator
    def gen_cell(self):
        return self.decoder.rnn.cell.gen_cell

    def factors_of(self, G):
        return self.decoder.rnn.cell.fac_linear(G)

    @torch.no_grad()
    def run_generator(self, g0, U, t0=0):
        """Generator only, driven by a given controller-output sequence U (n,T,co).
        g0 is the state before step t0. Returns G (n,T,gen) with zeros before t0."""
        cell = self.gen_cell()
        n, T, _ = U.shape
        G = torch.zeros(n, T, self.hps.gen_dim)
        g = g0
        for t in range(t0, T):
            g = cell(U[:, t], g)
            G[:, t] = g
        return G

    @torch.no_grad()
    def run_closed_loop(self, g0, ci, t0=0, gen_prev=None, con_prev=None):
        """Full decoder (controller in the loop, posterior means) from step t0, using the
        given controller inputs ci (n,T,2*ci_enc). States before t0 are taken as given."""
        cell = self.decoder.rnn.cell
        hps = self.hps
        n, T, _ = ci.shape
        g = g0 if gen_prev is None else gen_prev
        c = torch.tile(self.decoder.con_h0, (n, 1)) if con_prev is None else con_prev
        f = cell.fac_linear(g)
        G = torch.zeros(n, T, hps.gen_dim); U = torch.zeros(n, T, hps.co_dim)
        for t in range(t0, T):
            c = cell.con_cell(torch.cat([ci[:, t], f], 1), c)
            co_mean, _ = torch.split(cell.co_linear(c), hps.co_dim, dim=1)
            g = cell.gen_cell(co_mean, g)
            f = cell.fac_linear(g)
            G[:, t] = g; U[:, t] = co_mean
        return G, U


def l2_penalty(m, scale_gen, scale_con):
    terms = [(m.decoder.rnn.cell.gen_cell.weight_hh, scale_gen),
             (m.decoder.rnn.cell.con_cell.weight_hh, scale_con)]
    pen, size = 0.0, 0
    for k, w in terms:
        if w > 0:
            pen = pen + w * 0.5 * torch.norm(k, 2) ** 2
            size += k.numel()
    return pen / (size + 1e-8)


def ramp(step, start, length):
    if length <= 0:
        return 1.0
    return float(np.clip((step - start) / length, 0.0, 1.0))


def split_idx(n, n_train, split_seed=123):
    """Fixed train/test split of trials (the same for every fit of one dataset)."""
    perm = np.random.default_rng(split_seed).permutation(n)
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def subset(data, idx):
    """Per-trial arrays restricted to trials idx; everything else passed through."""
    out = {}
    n = data["spikes"].shape[0]
    for k, val in data.items():
        if hasattr(val, "shape") and len(val.shape) >= 1 and val.shape[0] == n and k != "C":
            out[k] = val[torch.as_tensor(idx)] if torch.is_tensor(val) else val[idx]
        else:
            out[k] = val
    return out


def fit(data, kl_co=1e-4, kl_ic=1e-6, l2_gen=1e-4, l2_con=1e-4, steps=3000, batch=64,
        lr=4e-3, seed=0, cd_rate=0.3, co_dim=4, log_every=250, hps_kw=None, ic_win=0, train_idx=None,
        sample_co=True):
    torch.manual_seed(seed); np.random.seed(seed)
    S = data["spikes"]; nin = data["n_heldin"]
    X = S[..., :nin].float()
    if train_idx is not None:
        X = X[torch.as_tensor(train_idx)]
    n, T, _ = X.shape
    hps = make_hps(nin, T, co_dim=co_dim, ic_win=ic_win, **(hps_kw or {}))
    m = LFADS(hps)
    opt = torch.optim.Adam(m.parameters(), lr=lr, eps=1e-8)
    cd = CoordinatedDropout(cd_rate=cd_rate, cd_pass_rate=0.0, ic_enc_seq_len=hps.ic_enc_seq_len)
    rng = np.random.default_rng(seed)
    t_start = time.time(); hist = []
    for step in range(steps):
        m.train()
        idx = torch.from_numpy(rng.choice(n, batch, replace=False))
        x = X[idx]
        (x_in,) = cd.process_batch((x,))
        out = m(x_in, sample=True, sample_co=sample_co)
        rec = F.poisson_nll_loss(out["logr"], x, full=True, reduction="none")
        rec = cd.process_losses(rec).mean()
        kl_co_v = m.co_prior(out["co_mean"], out["co_std"])
        kl_ic_v = m.ic_prior(out["ic_mean"], out["ic_std"])
        r_kl = ramp(step, 0, steps // 5); r_l2 = ramp(step, 0, steps // 5)
        loss = 1e4 * (rec + r_l2 * l2_penalty(m, l2_gen, l2_con) + r_kl * (kl_co * kl_co_v + kl_ic * kl_ic_v))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 200.0)
        opt.step()
        if step > steps * 0.6:
            for gp in opt.param_groups:
                gp["lr"] = lr * (0.999 ** (step - int(steps * 0.6)))
        if step % log_every == 0 or step == steps - 1:
            hist.append(dict(step=step, rec=rec.item(), kl_co=kl_co_v.item(), kl_ic=kl_ic_v.item()))
            print(f"step {step:5d} rec {rec.item():.4f} kl_co {kl_co_v.item():8.2f} kl_ic {kl_ic_v.item():7.2f}"
                  f"  {time.time() - t_start:6.0f}s", flush=True)
    m.eval()
    return m, hist


@torch.no_grad()
def infer(m, data):
    m.eval()
    X = data["spikes"][..., :data["n_heldin"]].float()
    return m(X, sample=False)


def glm_weights(X, Y, steps=400, l2=1e-3):
    Xf = X.reshape(-1, X.shape[-1]); Yf = Y.reshape(-1, Y.shape[-1])
    Wt = torch.zeros(X.shape[-1], Y.shape[-1], requires_grad=True)
    bt = torch.log(Yf.mean(0).clamp(min=1e-3)).clone().requires_grad_(True)
    o = torch.optim.LBFGS([Wt, bt], max_iter=steps, line_search_fn="strong_wolfe")
    def closure():
        o.zero_grad()
        lr = Xf @ Wt + bt
        l = (torch.exp(lr) - Yf * lr).mean() + l2 * (Wt ** 2).sum()
        l.backward(); return l
    o.step(closure)
    return Wt.detach(), bt.detach()


@torch.no_grad()
def evaluate_split(m, data, tr, te):
    """Held-in and co-smoothing scores on test trials; the co-smoothing GLM is fitted on
    training trials only."""
    S = data["spikes"]; nin = data["n_heldin"]
    o = infer(m, data)
    out = {}
    tr_t, te_t = torch.as_tensor(tr), torch.as_tensor(te)
    with torch.enable_grad():
        Wg, bg = glm_weights(o["factors"][tr_t], S[tr_t][..., nin:])
    for name, ix in (("train", tr_t), ("test", te_t)):
        out[f"bps_heldin_{name}"] = bits_per_spike(o["logr"][ix], S[ix][..., :nin])
        out[f"bps_cosmooth_{name}"] = bits_per_spike(o["factors"][ix] @ Wg + bg, S[ix][..., nin:])
        R = torch.as_tensor(data["rates"][..., :nin]).float()[ix]
        pr = torch.exp(o["logr"][ix])
        out[f"rate_r2_{name}"] = float(1 - ((pr - R) ** 2).sum() / ((R - R.mean((0, 1))) ** 2).sum())
    return out


@torch.no_grad()
def evaluate(m, data):
    S = data["spikes"]; nin = data["n_heldin"]
    o = infer(m, data)
    out = dict(bps_heldin=bits_per_spike(o["logr"], S[..., :nin]))
    with torch.enable_grad():
        lr_out = fit_glm(o["factors"], S[..., nin:])
    out["bps_cosmooth"] = bits_per_spike(lr_out, S[..., nin:])
    # rate fit to the true rates (teacher), held-in neurons
    if "rates" in data:
        R = torch.as_tensor(data["rates"][..., :nin]).float()
        pr = torch.exp(o["logr"])
        out["rate_r2"] = float(1 - ((pr - R) ** 2).sum() / ((R - R.mean((0, 1))) ** 2).sum())
    # autonomy: generator from the same g0 with co = 0
    G = o["gen"]
    Ga = m.run_generator(o["g0"], torch.zeros_like(o["co_mean"]))
    Gf = G.reshape(-1, G.shape[-1]); Gaf = Ga.reshape(-1, G.shape[-1])
    out["autonomy"] = float(1 - ((Gf - Gaf) ** 2).sum() / ((Gf - Gf.mean(0)) ** 2).sum())
    U = o["co_mean"]
    out["co_power"] = float((U ** 2).mean())
    Un = U.norm(dim=-1).numpy()
    try:
        out["u_go_locked"] = np.stack([Un[i, tg - 20:tg + 40] for i, tg in enumerate(data["t_go"])]).mean(0).tolist()
        out["u_load_locked"] = np.stack([Un[i, tl - 5:tl + 25] for i, tl in enumerate(data["t_load"])]).mean(0).tolist()
    except Exception:
        pass
    # open-loop replay check: generator driven by recorded co means reproduces G
    Gr = m.run_generator(o["g0"], U)
    out["replay_err"] = float((Gr - G).abs().max())
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/dataset.pt")
    p.add_argument("--tag", default="A")
    p.add_argument("--kl_co", type=float, default=1e-4)
    p.add_argument("--l2_gen", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--co_dim", type=int, default=4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--ic_win", type=int, default=0, help="IC encoder sees only the first ic_win bins")
    p.add_argument("--n_train", type=int, default=0, help="train on this many trials, test on the rest (0 = all)")
    p.add_argument("--no_co_sample", action="store_true", help="control: pass the controller mean, not a sample, in training")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    data = torch.load(a.data, weights_only=False)
    name = f"lfads_{a.tag}_kl{a.kl_co:g}_s{a.seed}" + ("" if a.co_dim == 4 else f"_co{a.co_dim}") + \
        (f"_tr{a.n_train}" if a.n_train else "") + ("_nosamp" if a.no_co_sample else "")
    print(name, flush=True)
    tr, te = split_idx(data["spikes"].shape[0], a.n_train) if a.n_train else (None, None)
    m, hist = fit(data, kl_co=a.kl_co, l2_gen=a.l2_gen, steps=a.steps, batch=a.batch, seed=a.seed, co_dim=a.co_dim,
                  ic_win=a.ic_win, train_idx=tr, sample_co=not a.no_co_sample)
    ev = evaluate(m, data)
    if a.n_train:
        ev.update(evaluate_split(m, data, tr, te))
        ev["train_idx"] = tr.tolist(); ev["test_idx"] = te.tolist()
    print(json.dumps({k: v for k, v in ev.items() if not isinstance(v, list)}, indent=1), flush=True)
    hp = {k: v for k, v in vars(m.hps).items() if k != "co_prior"}
    torch.save(dict(state_dict=m.state_dict(), hps=hp, kl_co=a.kl_co, l2_gen=a.l2_gen, seed=a.seed,
                    data=a.data, eval=ev, hist=hist), f"runs/{name}.pt")
