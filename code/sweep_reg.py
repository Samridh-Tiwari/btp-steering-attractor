"""
Does regularisation decide whether the network builds a clean fixed-point
manifold or a chaotic reservoir?

WHY THIS SWEEP
--------------
The first integrator network (g=1.0, reg_h=1e-4) solved the task and held
arbitrary values indefinitely -- but its autonomous dynamics never came to
rest. After 2000 silent steps the state was still moving ~6.5 per step, with
||h|| ~ 6.35, wandering over a region as large as itself, while the readout
stayed constant to three decimals. That is a continuum of CHAOTIC attracting
sets indexed by the memory value, not a line of fixed points.

That distinction matters for this project rather than being a curiosity. The
controllability machinery -- linearise, take the Jacobian, build the Gramian --
assumes a fixed point to linearise around. There isn't one. So before any
controllability measurement, the network has to be pushed into a regime where a
fixed-point manifold exists.

Two knobs plausibly control this:
  reg_h : L2 penalty on hidden activity. Larger -> smaller ||h|| -> the network
          sits in the near-linear part of tanh -> chaos is suppressed.
  g     : spectral scale at initialisation. g=1 starts at the edge of chaos;
          g<1 starts in the contracting regime.

This is the same axis as the cfg01/cfg06 split in the SURF work, where
regularisation strength decided how much of the solution lived in genuine
recurrent dynamics. Worth knowing whether it controls the attractor TYPE too.

WHAT IS MEASURED
----------------
  task MSE       -- can it still do the job
  hold error     -- does it still hold values through the silence
  settled speed  -- ||f(h) - h|| after long autonomous relaxation.
                    ~0     => genuine fixed-point manifold (what we want)
                    large  => chaotic / wandering attractor
  ||h||          -- how deep into tanh saturation the network sits
"""

import itertools
import torch

from train import train
from tasks import integrator
from model import VanillaRNN


def probe(net, T=200, batch=128, settle=2000, seed=3):
    torch.manual_seed(seed)
    with torch.no_grad():
        u, z, _ = integrator(batch=batch, T=T)
        y, H = net(u)
        hold_err = (y[:, -1, 0] - z[:, -1, 0]).abs().mean().item()
        spread = z[:, -1, 0].std().item()

        h = H[:, -1].clone()
        for _ in range(settle):
            h = net.f_autonomous(h)
        speed = (net.f_autonomous(h) - h).norm(dim=1).median().item()
        hnorm = h.norm(dim=1).mean().item()

        # does the continuum survive relaxation?
        r = net.readout(h).squeeze(-1)
        r_std = r.std().item()
    return dict(hold_err=hold_err, target_spread=spread, speed=speed,
                hnorm=hnorm, readout_std=r_std)


if __name__ == "__main__":
    grid = list(itertools.product([1e-4, 1e-3, 1e-2], [1.0, 0.7]))
    rows = []
    for reg_h, g in grid:
        print(f"\n=== reg_h={reg_h:g}  g={g} ===")
        net, cfg, mse, _ = train(task="integrator", steps=4000, T=200,
                                 reg_h=reg_h, g=g, seed=0, log_every=2000,
                                 outdir="runs")
        p = probe(net)
        p.update(reg_h=reg_h, g=g, test_mse=mse)
        rows.append(p)
        print(f"  hold err {p['hold_err']:.4f}   settled speed {p['speed']:.3e}   "
              f"||h|| {p['hnorm']:.2f}   readout std {p['readout_std']:.3f}")

    print("\n\n" + "=" * 92)
    print(f"{'reg_h':>8} {'g':>5} {'test MSE':>10} {'hold err':>9} "
          f"{'settled speed':>14} {'||h||':>7} {'readout std':>12}   verdict")
    print("=" * 92)
    for r in rows:
        if r["speed"] < 1e-3:
            v = "fixed-point manifold"
        elif r["speed"] < 1e-1:
            v = "slow / near-fixed"
        else:
            v = "chaotic / wandering"
        if r["readout_std"] < 0.05:
            v += ", continuum LOST"
        print(f"{r['reg_h']:8.0e} {r['g']:5.1f} {r['test_mse']:10.5f} {r['hold_err']:9.4f} "
              f"{r['speed']:14.3e} {r['hnorm']:7.2f} {r['readout_std']:12.3f}   {v}")
    print("=" * 92)
    print("want: low hold err (holds the value), low settled speed (fixed points exist),")
    print("      and readout std near the target spread (~0.45, continuum preserved)")
