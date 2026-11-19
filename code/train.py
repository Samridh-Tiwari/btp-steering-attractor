"""
Training loop.

Kept boring on purpose. The interesting part of this project is the analysis,
so training needs to be reproducible and unremarkable rather than clever.

Loss = masked MSE on the readout + an L2 penalty on the hidden state.

The hidden-state penalty (reg_h) is not decoration. It is the closest analogue
here of the LFADS regularisation knob that produced the cfg01-vs-cfg06 split in
the SURF work: it pushes the network toward low-amplitude states, which changes
how much of the solution lives in genuine recurrent dynamics versus in the
input drive. It is one of the axes to sweep.

Usage:
    python train.py --task flipflop --steps 3000
"""

import argparse
import json
import os
import torch

from model import VanillaRNN
from tasks import TASKS


def masked_mse(y, z, m):
    return ((y - z) ** 2 * m).sum() / m.sum().clamp(min=1.0)


def train(task="flipflop", n_hid=100, steps=3000, batch=64, T=100, lr=1e-3,
          g=1.0, input_scale=1.0, reg_h=1e-4, seed=0, device="cpu",
          log_every=250, outdir="runs"):

    torch.manual_seed(seed)
    task_fn = TASKS[task]

    # probe the task once to learn its input/output widths
    u0, z0, _ = task_fn(batch=2, T=8, device=device)
    n_in, n_out = u0.shape[-1], z0.shape[-1]

    net = VanillaRNN(n_in, n_hid, n_out, g=g, input_scale=input_scale, seed=seed).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    history = []
    for it in range(1, steps + 1):
        u, z, m = task_fn(batch=batch, T=T, device=device)
        y, H = net(u)
        task_loss = masked_mse(y, z, m)
        loss = task_loss + reg_h * (H ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if it % log_every == 0 or it == 1:
            history.append({"step": it, "task_loss": task_loss.item()})
            print(f"  step {it:5d}   task MSE {task_loss.item():.5f}")

    # held-out evaluation on fresh trials
    with torch.no_grad():
        u, z, m = task_fn(batch=256, T=T, device=device)
        y, H = net(u)
        test_mse = masked_mse(y, z, m).item()

        # task-appropriate accuracy: for flipflop, fraction of masked timesteps
        # where sign(output) matches the target bit.
        acc = None
        if task == "flipflop":
            correct = ((torch.sign(y) == z) * m).sum() / m.sum()
            acc = correct.item()

    print(f"\n  held-out MSE {test_mse:.5f}" + (f"   sign accuracy {acc:.4f}" if acc is not None else ""))

    cfg = dict(task=task, n_hid=n_hid, steps=steps, batch=batch, T=T, lr=lr, g=g,
               input_scale=input_scale, reg_h=reg_h, seed=seed)
    os.makedirs(outdir, exist_ok=True)
    tag = f"{task}_h{n_hid}_g{g}_in{input_scale}_reg{reg_h}_s{seed}"
    torch.save({"state_dict": net.state_dict(), "cfg": cfg,
                "test_mse": test_mse, "acc": acc, "history": history},
               os.path.join(outdir, tag + ".pt"))
    with open(os.path.join(outdir, tag + ".json"), "w") as fh:
        json.dump({"cfg": cfg, "test_mse": test_mse, "acc": acc, "history": history}, fh, indent=2)
    print(f"  saved runs/{tag}.pt")

    return net, cfg, test_mse, acc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="flipflop", choices=list(TASKS))
    p.add_argument("--n_hid", type=int, default=100)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--g", type=float, default=1.0)
    p.add_argument("--input_scale", type=float, default=1.0)
    p.add_argument("--reg_h", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(f"training {a.task}  N={a.n_hid}  g={a.g}  input_scale={a.input_scale}  reg_h={a.reg_h}  seed={a.seed}")
    train(**vars(a))
