"""Figure 12: can the model tell an internal memory from an outside input?"""
import json
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F
F.apply()
d = json.load(open("runs/split_study.json")); T = d["truth"]; fits = d["fits"]
cA, cB = F.OPT, F.TEST
fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.6), gridspec_kw=dict(wspace=0.38))
lams = sorted({f["lam"] for f in fits})
def jit(f): return np.log10(f["lam"]) + (f["seed"] - 1) * 0.12 + (0.05 if f["teacher"] == "B" else -0.05)
ax = axs[0]
for f in fits:
    ax.scatter(jit(f), f["r_silenced"], s=24, color=cA if f["teacher"] == "A" else cB, zorder=3)
ax.axhline(T["A"]["r_silenced"], color=cA, ls="--", lw=1.4); ax.axhline(T["B"]["r_silenced"], color=cB, ls="--", lw=1.4)
ax.text(-4.3, T["A"]["r_silenced"] + 0.03, "truth A: plan kept", color=cA, fontsize=7.5, ha="left")
ax.text(-4.3, T["B"]["r_silenced"] + 0.03, "truth B: plan lost", color=cB, fontsize=7.5, ha="left")
ax.set_xticks(np.log10(lams)); ax.set_xticklabels([f"{l:g}" for l in lams]); ax.set_xlabel("controller penalty")
ax.set_ylabel("pull vs plan with inputs off (r)"); ax.set_ylim(-0.6, 1.1)
ax.set_title("What happens if the inputs are switched off?", fontsize=9); F.panel_label(ax, "a")
ax = axs[1]
for f in fits:
    ax.scatter(jit(f), f["slow_eigs"][0], s=24, color=cA if f["teacher"] == "A" else cB, zorder=3)
ax.axhline(T["A"]["slow_eigs"][0], color=cA, ls="--", lw=1.4); ax.axhline(T["B"]["slow_eigs"][0], color=cB, ls="--", lw=1.4)
ax.text(1.15, T["A"]["slow_eigs"][0] + 0.012, "truth A", color=cA, fontsize=7.5, ha="right")
ax.text(1.15, T["B"]["slow_eigs"][0] + 0.012, "truth B", color=cB, fontsize=7.5, ha="right")
ax.set_xticks(np.log10(lams)); ax.set_xticklabels([f"{l:g}" for l in lams]); ax.set_xlabel("controller penalty")
ax.set_ylabel("slowest eigenvalue in the delay"); ax.set_ylim(0.6, 1.08)
ax.set_title("Does the model find a slow direction?", fontsize=9); F.panel_label(ax, "b")
ax = axs[2]
B = [f for f in fits if f["teacher"] == "B"]
ax.scatter([f["cosmooth"] for f in B], [f["r_silenced"] for f in B], s=26, color=cB, zorder=3)
ax.axhline(T["B"]["r_silenced"], color=cB, ls="--", lw=1.4)
ax.set_xlabel("held-out neuron prediction (bits/spike)"); ax.set_ylabel("predicted r with inputs off")
ax.set_title("Teacher B: the best models tell the wrong story", fontsize=9); F.panel_label(ax, "c")
ax.set_ylim(-0.6, 1.1)
h = [plt.Line2D([], [], marker="o", ls="", color=cA, label="students of teacher A (internal attractor)"),
     plt.Line2D([], [], marker="o", ls="", color=cB, label="students of teacher B (outside input)"),
     plt.Line2D([], [], ls="--", color=F.INK3, label="truth, measured on the teacher")]
fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.14))
fig.savefig("figs/fig12_split.png", bbox_inches="tight"); plt.close(fig)
print("ok")
