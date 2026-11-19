"""Figure 10: the steering question on public monkey data (MC_Maze large)."""
import glob, json, re
import numpy as np
import torch
import matplotlib.pyplot as plt
import figstyle as F

F.apply()
data = torch.load("runs/mc_data.pt", weights_only=False)
rows = []
for p in sorted(glob.glob("runs/mc_student_k*_steer.json")):
    r = json.load(open(p))
    if r["eval"]["bps_heldin"] < 0:
        print("skip diverged", p); continue
    br = r["by_ridge"]["0.001"]
    rows.append(dict(k=r["k"], lam=r["lam_u"], seed=r["seed"], ev=r["eval"], vel_r2=br["reach_r2"],
                     d=br["delay"]["fit"], g=br["go"]["fit"], path=p,
                     spread=float(np.ptp([r["by_ridge"][x]["delay"]["fit"]["frac_full"]["median"]
                                          for x in r["by_ridge"]]))))
rows.sort(key=lambda r: (r["k"], r["lam"], r["seed"]))

fig = plt.figure(figsize=(10.6, 4.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.42)

# a: every fit, delay and go windows, own vs random vs best
ax = fig.add_subplot(gs[0])
labels = []
for i, r in enumerate(rows):
    y = len(rows) - 1 - i
    for dy, win, mk in ((0.17, r["d"], "s"), (-0.17, r["g"], "o")):
        rnd = r["k"] / 64
        ax.plot([rnd * 100, win["frac_full_opt"]["median"] * 100], [y + dy] * 2, color=F.GRID, lw=5,
                solid_capstyle="round", zorder=1)
        ax.plot(rnd * 100, y + dy, "|", color=F.RAND, ms=9, mew=2.2, zorder=2)
        ax.plot(win["frac_full_opt"]["median"] * 100, y + dy, "|", color=F.OPT, ms=9, mew=2.2, zorder=2)
        ax.plot(win["frac_full"]["median"] * 100, y + dy, mk, color=F.TEST, ms=5.5, zorder=3)
    labels.append(f"k={r['k']}, penalty {r['lam']:g}, seed {r['seed']}")
ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels[::-1], fontsize=7)
ax.set_xscale("log"); ax.set_xlim(1, 110); ax.set_xlabel("fraction of full authority over the reach endpoint (%)")
ax.set_title("Monkey motor cortex: inferred channels\n(square: last 300 ms of delay, circle: around go)", fontsize=9)
F.panel_label(ax, "a")

# b: generalisation vs aim
ax = fig.add_subplot(gs[1])
for r in rows:
    c = F.TEST if r["k"] == 4 else F.INK3
    ax.scatter(r["ev"]["bps_cosmooth"], r["d"]["frac_full"]["median"] * 100, s=26, color=c, zorder=3)
    if r["k"] != 4: ax.annotate(f"k = {r['k']}, seed {r['seed']}", (r["ev"]["bps_cosmooth"], r["d"]["frac_full"]["median"] * 100),
                xytext=(-4, 6), textcoords="offset points", fontsize=6, ha="right")
ax.set_xlabel("held-out neuron prediction (bits/spike)"); ax.set_ylabel("aim during the delay (%)")
ax.text(0.97, 0.97, "orange: k = 4, both penalties", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=F.TEST)
ax.set_title("Better models are not better aimed", fontsize=9); F.panel_label(ax, "b")

fig.savefig("figs/fig10_mc_maze.png"); plt.close(fig)

for r in rows:
    print(f"k{r['k']} lam {r['lam']:g} s{r['seed']}: cos {r['ev']['bps_cosmooth']:.3f} heldin {r['ev']['bps_heldin']:.3f} "
          f"auto {r['ev']['autonomy']:.2f} reachR2 {np.round(r['vel_r2'], 2)} | delay own {r['d']['frac_full']['median']*100:.1f} "
          f"best {r['d']['frac_full_opt']['median']*100:.1f} rand {r['k']/64*100:.1f} x{r['d']['ratio_rand']['median']:.2f} "
          f"pct {r['d']['pct']['median']:.0f} | go own {r['g']['frac_full']['median']*100:.1f} best {r['g']['frac_full_opt']['median']*100:.1f} "
          f"| ridge spread {r['spread']*100:.1f}")
