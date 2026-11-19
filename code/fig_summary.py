"""One picture of the whole result: own channels vs random vs best, in all three systems."""
import matplotlib.pyplot as plt
import numpy as np
import figstyle as F
F.apply()
rows = [  # label, random, own values, best values
    ("Teacher: its real inputs\n(2 channels)", 2.0, [15.0], [100.0]),
    ("Simplified student, weak penalty\n(4 inferred channels, 4 seeds)", 6.25, [23.9, 22.3, 41.6, 28.2], [56.0, 57.0, 66.7, 55.9]),
    ("Simplified student, strong penalty\n(4 inferred channels, 4 seeds)", 6.25, [15.3, 21.6, 51.0, 32.7], [96.5, 92.7, 99.8, 96.4]),
    ("Monkey, simplified, weak penalty\n(4 inferred channels, 3 seeds)", 6.25, [4.2, 5.4, 8.0], [92.0, 89.3, 83.8]),
    ("Monkey, simplified, strong penalty\n(4 inferred channels, 3 seeds)", 6.25, [17.8, 10.2, 11.7], [97.1, 98.4, 97.9]),
]
import json, re
LT = json.load(open("runs/lfads_tests.json")); MC = json.load(open("runs/lfads_mc_tests.json"))
fa = [r for p, r in LT.items() if re.match(r"runs/lfads_Ac?_kl[0-9.e-]+_s\d_tr192(_nosamp)?\.pt$", p)]
rows.insert(3, (f"Full LFADS, teacher A\n(4 inferred channels, {len(fa)} fits)", 6.25,
                [100 * r["authority"]["frac_full"] for r in fa], [100 * r["authority"]["frac_opt"] for r in fa]))
rows.append((f"Full LFADS, monkey\n(4 inferred channels, {len(MC)} fits)", 6.25,
             [100 * r["authority"]["frac_full"] for r in MC.values()], [100 * r["authority"]["frac_opt"] for r in MC.values()]))
fig, ax = plt.subplots(figsize=(10.0, 5.4))
for i, (lab, rnd, own, best) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([0, 100], [y, y], color=F.GRID, lw=10, solid_capstyle="round", zorder=0)
    ax.plot(rnd, y, "|", color=F.RAND, ms=18, mew=3, zorder=2)
    ax.plot(best, [y] * len(best), "o", color=F.OPT, ms=7, zorder=3)
    ax.plot(own, [y] * len(own), "o", color=F.TEST, ms=7, zorder=4)
    if max(own) < 1:
        txt = f"own: {min(own):.2f}–{max(own):.1f}%"
    else:
        txt = f"own: {min(own):.0f}–{max(own):.0f}%" if len(own) > 1 else f"real inputs: {own[0]:.0f}%"
    ax.text(max(np.mean(own), 6), y + 0.32, txt, ha="center", fontsize=8, color=F.INK2)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
ax.set_xlim(-2, 103); ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("authority over behaviour, as % of driving every unit")
ax.plot([], [], "|", color=F.RAND, ms=12, mew=3, label="random channels (same number)")
ax.plot([], [], "o", color=F.TEST, label="the model's own inferred channels")
ax.plot([], [], "o", color=F.OPT, label="best possible channels (same number)")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, fontsize=8, frameon=False)
ax.set_title("In every system the best channels steer almost fully; the inferred ones do not", fontsize=10)
fig.savefig("figs/fig11_summary.png"); plt.close(fig)
print("ok")
