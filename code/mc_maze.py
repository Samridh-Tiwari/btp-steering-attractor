"""
The SURF steering question on PUBLIC data: Neural Latents Benchmark MC_Maze
(monkey Jenkins, PMd/M1 Utah arrays, delayed centre-out-and-maze reaching).

Same pipeline as the synthetic students, no ground truth this time:
  1. bin spikes at 20 ms, align trials to the go cue, keep trials whose delay
     is long enough to contain the analysis window
  2. fit the LFADS-style student (lfads_lite.Student) to held-in neurons;
     co-smoothing on held-out neurons
  3. behaviour = the reach: hand displacement over the first 400 ms after
     movement onset (x and y), read out from the student's generator with a
     ridge decoder to hand velocity (lagged, M1 leads the hand)
  4. causal sensitivity of the reach to an added input during the last 300 ms
     of the delay, back-propagated through the student's own dynamics
  5. the student's inferred channels vs random and best channels of the same k

    python mc_maze.py inspect <file.nwb>
    python mc_maze.py prep    <file.nwb>            -> runs/mc_data.pt
    python mc_maze.py fit     --k 4 --lam_u 0.1 --seed 0
    python mc_maze.py steer   runs/mc_student_*.pt
"""
import argparse
import glob
import json
import sys
import numpy as np
import torch

from lfads_lite import Student, fit_student, bits_per_spike, fit_glm
import steer as S

BIN = 0.02            # s
PRE_GO = 0.30         # s of delay inside the window
POST_GO = 0.80        # s after go
LAG = 5               # bins (100 ms): generator state at t -> hand velocity at t+LAG
REACH_BINS = 20       # 400 ms of movement define the reach


# --------------------------------------------------------------------------- data
def inspect(path):
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(path, "r") as io:
        f = io.read()
        print("units:", len(f.units), "columns", f.units.colnames)
        print("trials:", len(f.trials), "columns", f.trials.colnames)
        for name, mod in f.processing.items():
            print("processing", name, list(mod.data_interfaces))
        print("acquisition", list(f.acquisition))
        tr = f.trials.to_dataframe()
        print(tr.head(3).T)
        if "split" in tr:
            print(tr["split"].value_counts())


def prep(path, out="runs/mc_data.pt", heldout_frac=0.2, seed=0):
    from pynwb import NWBHDF5IO
    rng = np.random.default_rng(seed)
    with NWBHDF5IO(path, "r") as io:
        f = io.read()
        tr = f.trials.to_dataframe()
        units = f.units.to_dataframe()
        beh = f.processing["behavior"]
        hv = beh.data_interfaces["hand_vel"]
        vel = np.asarray(hv.data[:])
        vt = np.asarray(hv.timestamps[:]) if hv.timestamps is not None else \
            hv.starting_time + np.arange(len(vel)) / hv.rate
        spikes = [np.asarray(s) for s in units["spike_times"]]
        heldout = np.asarray(units["heldout"]) if "heldout" in units else None

    keep = np.ones(len(tr), bool)
    if "split" in tr:
        keep &= tr["split"].isin(["train", "val"]).to_numpy()
    if "success" in tr:
        keep &= tr["success"].astype(bool).to_numpy()
    delay = (tr["go_cue_time"] - tr["target_on_time"]).to_numpy()
    keep &= delay >= PRE_GO + 0.05
    keep &= np.isfinite(tr["move_onset_time"].to_numpy())
    tr = tr[keep].reset_index(drop=True)

    nb_pre, nb_post = int(round(PRE_GO / BIN)), int(round(POST_GO / BIN))
    T = nb_pre + nb_post
    edges0 = np.arange(-nb_pre, nb_post + 1) * BIN
    X = np.zeros((len(tr), T, len(spikes)), np.float32)
    V = np.zeros((len(tr), T, 2), np.float32)
    t_move = np.zeros(len(tr), int)
    for i, row in tr.iterrows():
        e = row["go_cue_time"] + edges0
        for j, st in enumerate(spikes):
            X[i, :, j] = np.histogram(st, e)[0]
        c = e[:-1] + BIN / 2
        V[i, :, 0] = np.interp(c, vt, vel[:, 0]); V[i, :, 1] = np.interp(c, vt, vel[:, 1])
        t_move[i] = nb_pre + int(round((row["move_onset_time"] - row["go_cue_time"]) / BIN))

    ok = t_move + REACH_BINS <= T
    X, V, t_move = X[ok], V[ok], t_move[ok]
    tr = tr[ok].reset_index(drop=True)

    # drop near-silent units, then order held-in first, held-out last
    rate = X.mean((0, 1)) / BIN
    good = rate > 1.0
    if heldout is not None and heldout[good].any():
        ho = heldout[good].astype(bool)
    else:
        ho = rng.random(good.sum()) < heldout_frac
    Xg = X[..., good]
    order = np.r_[np.where(~ho)[0], np.where(ho)[0]]
    Xg = Xg[..., order]
    reach = V[np.arange(len(V))[:, None], t_move[:, None] + np.arange(REACH_BINS)].sum(1) * BIN
    data = dict(spikes=torch.from_numpy(Xg), vel=V, t_move=t_move,
                t_go=np.full(len(Xg), nb_pre), t_load=np.zeros(len(Xg), int),
                n_heldin=int((~ho).sum()), reach=reach,
                delay=(tr["go_cue_time"] - tr["target_on_time"]).to_numpy(),
                cond=tr["trial_type"].to_numpy() if "trial_type" in tr else None,
                bin=BIN, pre_go_bins=nb_pre)
    torch.save(data, out)
    print(f"{len(Xg)} trials, T={T} bins, {Xg.shape[-1]} units "
          f"({data['n_heldin']} held-in, {int(ho.sum())} held-out), "
          f"mean rate {Xg.mean() / BIN:.1f} Hz, reach length median "
          f"{np.median(np.linalg.norm(reach, axis=1)):.1f}")
    return data


# --------------------------------------------------------------------------- fit
def evaluate_mc(st, data):
    S_ = data["spikes"]; nin = data["n_heldin"]
    with torch.no_grad():
        G, _ = st.generate(use_u=True)
        Ga, _ = st.generate(use_u=False)
        fac = G @ st.F.T
        out = dict(bps_heldin=bits_per_spike(st.logrates(G).clamp(max=2.5), S_[..., :nin]))
    out["bps_cosmooth"] = bits_per_spike(fit_glm(fac.detach(), S_[..., nin:]), S_[..., nin:])
    a = G.reshape(-1, st.N); b = Ga.reshape(-1, st.N)
    out["autonomy"] = 1 - ((a - b) ** 2).sum().item() / ((a - a.mean(0)) ** 2).sum().item()
    out["u_power"] = (st.U ** 2).mean().item()
    return out


def fit_student_stable(data, k=4, lam_u=0.1, N=64, steps=2500, lr=5e-3, seed=0,
                       lam_w=1e-5, log_every=250, max_logr=2.5):
    """lfads_lite.fit_student with the log-rate capped (exp(2.5) = 12 spikes per bin)
    and a smaller learning rate: real data diverged at the synthetic settings."""
    import time
    from lfads_lite import poisson_nll
    torch.manual_seed(seed)
    Sin = data["spikes"][..., :data["n_heldin"]]
    n, T, nin = Sin.shape
    st = Student(n, T, nin, N=N, k=k, seed=seed, mean_rate=Sin.mean(dim=(0, 1)).clamp(min=1e-3))
    opt = torch.optim.Adam(st.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    t0 = time.time()
    for it in range(1, steps + 1):
        G, _ = st.generate()
        lr_ = st.logrates(G).clamp(max=max_logr)
        nll = poisson_nll(lr_, Sin)
        loss = nll + lam_u * (st.U ** 2).mean() + lam_w * (st.W ** 2).sum() + 1e-3 * (st.g0 ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0)
        opt.step(); sched.step()
        if it % log_every == 0 or it == 1:
            print(f"  step {it:5d}  nll {nll.item():.4f}  |u|^2 {(st.U ** 2).mean().item():.4f}"
                  f"  bps {bits_per_spike(lr_.detach(), Sin):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return st


def fit(k, lam_u, seed, steps, data_path="runs/mc_data.pt"):
    torch.set_num_threads(1)
    data = torch.load(data_path, weights_only=False)
    tag = f"mc_student_k{k}_lam{lam_u}_s{seed}"
    print(tag, flush=True)
    st = fit_student_stable(data, k=k, lam_u=lam_u, steps=steps, seed=seed)
    ev = evaluate_mc(st, data)
    print(json.dumps(ev, indent=1), flush=True)
    torch.save({"state_dict": st.state_dict(), "k": k, "lam_u": lam_u, "seed": seed,
                "N": st.N, "eval": ev}, f"runs/{tag}.pt")


# --------------------------------------------------------------------------- steer
@torch.no_grad()
def vel_decoder(st, data, ridge):
    G, _ = st.generate(use_u=True)
    G = G.numpy(); V = data["vel"]
    X = G[:, :-LAG].reshape(-1, st.N); Y = V[:, LAG:].reshape(-1, 2)
    Xm, Ym = X.mean(0), Y.mean(0)
    Xc = X - Xm
    W = np.linalg.solve(Xc.T @ Xc + ridge * len(X) * np.eye(st.N), Xc.T @ (Y - Ym))
    r2 = 1 - ((Xc @ W - (Y - Ym)) ** 2).sum(0) / ((Y - Ym) ** 2).sum(0)
    return torch.tensor(W, dtype=torch.float32), torch.tensor(Ym - Xm @ W, dtype=torch.float32), r2


@torch.no_grad()
def reach_decoder(st, data, ridge, folds=5):
    """Reach endpoint (x, y) from the generator state averaged over the movement window,
    ridge across trials. Returns weights, bias and the cross-validated R²."""
    G, _ = st.generate(use_u=True); G = G.numpy()
    tm = data["t_move"]; R = data["reach"]
    Fm = np.stack([G[i, t - LAG: t - LAG + REACH_BINS].mean(0) for i, t in enumerate(tm)])
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
    return torch.tensor(W, dtype=torch.float32), torch.tensor(b, dtype=torch.float32), r2


def reach_sensitivity(st, W, b, t_move):
    """s[c] (n, T, N): d reach_c / d x(t), c = x, y. Reach = sum of decoded velocity over the movement."""
    n, T, k = st.U.shape
    mask = torch.zeros(n, T)
    for i, tm in enumerate(t_move):
        mask[i, tm - LAG: tm - LAG + REACH_BINS] = 1.0 / REACH_BINS
    out = []
    for c in range(2):
        x = torch.zeros(n, T, st.N, requires_grad=True)
        g = torch.tanh(st.g0.detach()); U = st.U.detach(); Bm = st.B().detach()
        Wg = st.W.detach(); bg = st.b.detach()
        ys = []
        for t in range(T):
            g = torch.tanh(g @ Wg.T + bg + U[:, t] @ Bm.T + x[:, t])
            ys.append(g @ W[:, c] + b[c])
        amp = (torch.stack(ys, 1) * mask).sum(1)
        amp.sum().backward()
        out.append(x.grad.numpy())
    return out, None


def steer(paths, data_path="runs/mc_data.pt", ridges=(1e-4, 1e-3, 1e-2)):
    torch.set_num_threads(1)
    data = torch.load(data_path, weights_only=False)
    n, T, _ = data["spikes"].shape
    nb = data["pre_go_bins"]
    windows = {"delay": [list(range(0, nb - 2))] * n,                  # last 300 ms of the delay
               "go": [list(range(nb - 2, nb + 5))] * n}                # around the go cue
    for p in paths:
        ck = torch.load(p, weights_only=False)
        st = Student(n, T, data["n_heldin"], N=ck["N"], k=ck["k"])
        st.load_state_dict(ck["state_dict"]); st.eval()
        Bfit = st.B().detach().numpy()
        ck["eval"] = evaluate_mc(st, data)
        torch.save(ck, p)
        res = dict(path=p, k=ck["k"], lam_u=ck["lam_u"], seed=ck["seed"], eval=ck["eval"], by_ridge={})
        for ridge in ridges:
            W, b, r2 = reach_decoder(st, data, ridge)
            sx, _ = reach_sensitivity(st, W, b, data["t_move"])
            r = dict(reach_r2=r2.tolist())
            for name, idx in windows.items():
                Qs = [a + c for a, c in zip(S.Q_from(sx[0], idx), S.Q_from(sx[1], idx))]
                r[name] = S.compare(Qs, {"fit": Bfit})
            res["by_ridge"][str(ridge)] = r
        json.dump(res, open(p.replace(".pt", "_steer.json"), "w"), indent=1)
        m = res["by_ridge"]["0.001"]["delay"]["fit"]
        print(f"{p}: cosmooth {ck['eval']['bps_cosmooth']:.3f} autonomy {ck['eval']['autonomy']:.2f} "
              f"reach R2 {np.round(res['by_ridge']['0.001']['reach_r2'], 2)}  delay frac {m['frac_full']['median']:.3f} "
              f"best {m['frac_full_opt']['median']:.3f}  x rand {m['ratio_rand']['median']:.2f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inspect":
        inspect(sys.argv[2])
    elif cmd == "prep":
        prep(sys.argv[2])
    elif cmd == "fit":
        p = argparse.ArgumentParser()
        p.add_argument("--k", type=int, default=4); p.add_argument("--lam_u", type=float, default=0.1)
        p.add_argument("--seed", type=int, default=0); p.add_argument("--steps", type=int, default=2500)
        a = p.parse_args(sys.argv[2:])
        fit(a.k, a.lam_u, a.seed, a.steps)
    elif cmd == "steer":
        steer(sys.argv[2:] or sorted(glob.glob("runs/mc_student_*.pt")))
