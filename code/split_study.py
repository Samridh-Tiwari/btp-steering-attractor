"""
Can a fitted model tell an internal memory from an external input?

Teacher A  (existing):  value pulse, then silence; the plan is held by an internal line attractor.
Teacher B  (new):       the value input stays ON at amplitude v through the whole delay, until the
                        go cue; the network needs no memory, the plan is held from outside.

From spikes alone the two can look alike. We fit students to both over a range of controller
penalties and seeds, and ask:
  * which story the student tells: how much controller power it spends in the delay (input story)
    and how slow its own dynamics are (attractor story);
  * the ground-truth test: switch the inputs off during the delay. In A nothing changes (r stays
    ~1); in B the plan is lost. Does the student predict the right outcome?
  * whether held-out prediction (the usual way to choose a model) picks the student with the
    right story.

    python split_study.py train_b
    python split_study.py data_b
    python split_study.py truth
    python split_study.py analyse
"""
import glob
import json
import re
import sys
import numpy as np
import torch

from model import VanillaRNN
from teacher import pullshape, run_noisy, load_teacher, T_TRIAL, diagnose
import steer as S

DELAY_SIL = (8, 0)           # silence window used everywhere: t_load+8 .. t_go (go cue itself untouched)


def task(batch, rng, sustained, hint=False, return_meta=False, T=T_TRIAL):
    u = np.zeros((batch, T, 2), np.float32)
    z = np.zeros((batch, T, 2 if hint else 1), np.float32)
    v = rng.uniform(0.2, 1.0, batch)
    t_load = rng.integers(10, 20, batch)
    t_go = rng.integers(60, 100, batch)
    for b in range(batch):
        if sustained:
            u[b, t_load[b]:t_go[b], 0] = v[b]
        else:
            u[b, t_load[b]:t_load[b] + 5, 0] = v[b]
        u[b, t_go[b]:t_go[b] + 3, 1] = 1.0
        z[b, t_go[b]:, 0] = v[b] * pullshape(T - t_go[b])
        if hint:
            z[b, t_load[b] + 5:t_go[b], 1] = v[b]
    out = (torch.from_numpy(u), torch.from_numpy(z))
    return out + (dict(v=v, t_load=t_load, t_go=t_go),) if return_meta else out


def train_b(g=0.9, noise=0.05, n_hid=100, steps=6000, lr=1e-3, reg_h=1e-4, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = VanillaRNN(2, n_hid, 2, g=g, input_scale=1.0, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    for it in range(1, steps + 1):
        u, z = task(64, rng, sustained=True, hint=True)
        y, H = run_noisy(net, u, noise)
        l1 = ((y[..., 0] - z[..., 0]) ** 2).mean(); l2 = ((y[..., 1] - z[..., 1]) ** 2).mean()
        loss = l1 + 0.5 * l2 + reg_h * (H ** 2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 500 == 0 or it == 1:
            print(f"  step {it:5d}  lever mse {l1.item():.5f}", flush=True)
    torch.save({"state_dict": net.state_dict(), "cfg": dict(g=g, noise=noise, n_hid=n_hid, steps=steps, seed=seed)},
               "runs/teacherB_sustained_s0.pt")


@torch.no_grad()
def make_data_b(n_trials=256, n_obs=60, n_heldout=10, seed=7, mod_sd=0.7, rate_range=(0.08, 0.45)):
    """Same recipe as lfads_lite.make_dataset, with the sustained-input task."""
    net, _ = load_teacher("runs/teacherB_sustained_s0.pt")
    rng = np.random.default_rng(seed)
    u, z, meta = task(n_trials, rng, sustained=True, return_meta=True)
    _, H = run_noisy(net, u, 0.0); H = H.numpy()
    C = rng.standard_normal((n_obs, H.shape[-1]))
    X = H.reshape(-1, H.shape[-1]) @ C.T
    X = (X - X.mean(0)) / X.std(0)
    base = rng.uniform(*rate_range, n_obs)
    logr = np.log(base) + mod_sd * X - mod_sd ** 2 / 2
    rates = np.exp(logr).reshape(n_trials, -1, n_obs)
    spikes = rng.poisson(rates).astype(np.float32)
    torch.save(dict(spikes=torch.from_numpy(spikes), rates=rates.astype(np.float32), H_teacher=H.astype(np.float32),
                    u_teacher=u.numpy(), z=z.numpy(), v=meta["v"], t_go=meta["t_go"], t_load=meta["t_load"],
                    n_heldin=n_obs - n_heldout, C=C), "runs/dataset_B.pt")
    print("dataset B saved")


@torch.no_grad()
def teacher_truth(path, sustained, n=256, seed=11):
    """Ground truth: what happens to the pull when the teacher's inputs are switched off in the delay."""
    net, _ = load_teacher(path)
    rng = np.random.default_rng(seed)
    u, z, meta = task(n, rng, sustained=sustained, return_meta=True)
    tl, tg, v = meta["t_load"], meta["t_go"], meta["v"]
    mask = S.plateau_mask(tg, u.shape[1])
    def amp(uu):
        y, H = run_noisy(net, uu, 0.0)
        return (y[..., 0] * mask).sum(1).numpy(), H
    a_full, H = amp(u)
    u0 = u.clone()
    for i in range(n):
        u0[i, tl[i] + DELAY_SIL[0]:tg[i] + DELAY_SIL[1], 0] = 0.0     # plan input only
    a_sil, _ = amp(u0)
    # slowest eigenvalue mid-delay, with the inputs as they are in the trial
    W = net.W_rec.numpy(); Win = net.W_in.numpy(); b = net.b.numpy()
    lams = []
    for i in range(0, n, 8):
        t = tg[i] - 15
        pre = W @ H[i, t - 1].numpy() + Win @ u[i, t].numpy() + b
        A = (1 - np.tanh(pre) ** 2)[:, None] * W
        lams.append(np.sort(np.abs(np.linalg.eigvals(A)))[::-1][:3])
    return dict(r_full=float(np.corrcoef(a_full, v)[0, 1]), r_silenced=float(np.corrcoef(a_sil, v)[0, 1]),
                slow_eigs=np.median(lams, 0).tolist())


@torch.no_grad()
def silenced(st, data, w, b):
    """Student prediction: zero the controller over the same window as the teacher truth."""
    tl, tg, v = data["t_load"], data["t_go"], data["v"]
    mask = S.plateau_mask(tg, st.T)
    def amp(U):
        G, _ = st.generate(use_u=True, U=U)
        return ((G @ w + b) * mask).sum(1).numpy()
    U0 = st.U.detach().clone()
    for i in range(len(tg)):
        U0[i, tl[i] + DELAY_SIL[0]:tg[i] + DELAY_SIL[1]] = 0.0
    return dict(r_full=float(np.corrcoef(amp(st.U.detach()), v)[0, 1]), r_silenced=float(np.corrcoef(amp(U0), v)[0, 1]))


def analyse():
    from student_steer import load_student, lever_decoder, delay_silenced_fidelity, slow_eig_student
    out = {"truth": json.load(open("runs/split_truth.json")), "fits": []}
    for tag, data_path, pat in (("A", "runs/dataset.pt", r"runs/student_k4_lam([0-9.e-]+)_s(\d)\.pt$"),
                                ("B", "runs/dataset_B.pt", r"runs/student_k4_lam([0-9.e-]+)_s(\d)_B\.pt$")):
        data = torch.load(data_path, weights_only=False)
        for p in sorted(glob.glob("runs/student_k4_lam*_s*.pt")):
            m = re.match(pat, p)
            if not m or int(m.group(2)) > 2:
                continue
            st, ck = load_student(p, data)
            ev = ck["eval"]
            if not np.isfinite(ev["bps_heldin"]) or ev["bps_heldin"] < 0:
                print("skip diverged", p); continue
            w, b, r2 = lever_decoder(st, data, 1e-2)
            fid = silenced(st, data, w, b)
            U = st.U.detach().numpy() ** 2
            tl, tg = data["t_load"], data["t_go"]
            dmask = np.zeros(U.shape[:2], bool)
            for i in range(len(tl)):
                dmask[i, tl[i] + DELAY_SIL[0]:tg[i] + DELAY_SIL[1]] = True
            per_t = U.sum(-1)
            delay_share = float(per_t[dmask].sum() / per_t.sum())
            delay_power = float(per_t[dmask].mean())
            out["fits"].append(dict(teacher=tag, lam=float(m.group(1)), seed=int(m.group(2)),
                                    cosmooth=ev["bps_cosmooth"], heldin=ev["bps_heldin"],
                                    r_full=fid["r_full"], r_silenced=fid["r_silenced"],
                                    delay_share=delay_share, delay_power=delay_power,
                                    slow_eigs=slow_eig_student(st, data), lever_r2=r2))
            f = out["fits"][-1]
            print(f"{tag} lam {f['lam']:g} s{f['seed']}: cosmooth {f['cosmooth']:.3f}  silenced r {f['r_silenced']:.2f}"
                  f"  delay input share {f['delay_share']:.2f}  slow eig {f['slow_eigs'][0]:.3f}", flush=True)
    json.dump(out, open("runs/split_study.json", "w"), indent=1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    cmd = sys.argv[1]
    if cmd == "train_b":
        train_b()
        net, _ = load_teacher("runs/teacherB_sustained_s0.pt")
    elif cmd == "data_b":
        make_data_b()
    elif cmd == "truth":
        t = dict(A=teacher_truth("runs/teacher_g0.9_n0.05_s0.pt", sustained=False),
                 B=teacher_truth("runs/teacherB_sustained_s0.pt", sustained=True))
        json.dump(t, open("runs/split_truth.json", "w"), indent=1); print(json.dumps(t, indent=1))
    elif cmd == "analyse":
        analyse()
