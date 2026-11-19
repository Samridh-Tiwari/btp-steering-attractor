"""Figures for the plan-location tests.
figP1: where does a fitted model keep the plan (swap, shift, persistence, marginal count)
figP2: event students vs free students (held-out prediction, memory), and the monkey swap test"""
import json
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F
F.apply()
P = json.load(open("runs/plan_tests_ABT2.json")); FP = json.load(open("runs/fp_count.json")); EV = json.load(open("runs/event_tests.json"))
MC = json.load(open("runs/mc_plan_tests.json"))
cA, cB, cE = F.OPT, F.TEST, F.GO
lams = [1e-4, 1e-2, 0.1, 1.0, 10.0]
XE = 2.2                                    # x position (log10) for event students

def jit(f, tag): return np.log10(f["lam"]) + (f["seed"] - 2) * 0.07 + (0.06 if tag == "B" else -0.06)

def base(ax, ylabel, title, lab, event=True):
    xt = list(np.log10(lams)) + ([XE] if event else [])
    ax.set_xticks(xt); ax.set_xticklabels([f"{l:g}" for l in lams] + (["told\nwhen"] if event else []))
    ax.set_xlabel("controller penalty of the free student" + ("            " if event else ""))
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9); F.panel_label(ax, lab)
    if event: ax.axvline(1.6, color=F.GRID, lw=1.2)

fig, axs = plt.subplots(2, 2, figsize=(11.6, 7.4), gridspec_kw=dict(wspace=0.3, hspace=0.5))
# (a) swap
ax = axs[0, 0]
for tag, c in (("A", cA), ("B", cB)):
    for f in P[tag]["fits"]:
        ax.scatter(jit(f, tag), np.clip(f["swap_all"]["own"], -0.1, 1.1), s=22, color=c, zorder=3)
    ax.axhline(P[tag]["teacher"]["swap_all"]["own"], color=c, ls="--", lw=1.4)
ax.text(-4.3, 0.93, "truth A: the pull follows the trial's own plan", color=cA, fontsize=7.5)
ax.text(-4.3, -0.085, "truth B: it follows the swapped-in input", color=cB, fontsize=7.5)
ax.set_ylim(-0.12, 1.12)
base(ax, "share of the pull set by the trial's own state", "Swap test: give a trial another trial's inputs", "a", event=False)
# (b) shift
ax = axs[0, 1]
for tag, c in (("A", cA), ("B", cB)):
    for f in P[tag]["fits"]:
        ax.scatter(jit(f, tag), np.clip(f["shift_gain"], -0.3, 1.1), s=22, color=c, zorder=3)
    ax.axhline(P[tag]["teacher"]["shift_gain"], color=c, ls="--", lw=1.4)
    ev = [e["shift_gain"] for e in EV[tag]["event"] if abs(e["r_amp_v"]) > 0.5]
    ax.scatter(XE + np.linspace(-0.12, 0.12, len(ev)) + (0.0), ev, s=30, color=c, marker="D", zorder=3)
ax.set_ylim(-0.3, 1.12)
base(ax, "share of a plan shift that reaches the pull", "Shift test: nudge the state along the line of plans", "b")
# (c) persistence
ax = axs[1, 0]
for tag, c in (("A", cA), ("B", cB)):
    for f in P[tag]["fits"]:
        ax.scatter(jit(f, tag), np.clip(f["persist_realpos"], 0, 1.3), s=22, color=c, zorder=3)
    ax.axhline(P[tag]["teacher"]["persist_realpos"], color=c, ls="--", lw=1.4)
    ev = [e["persist_realpos"] for e in EV[tag]["event"]]
    ax.scatter(XE + np.linspace(-0.12, 0.12, len(ev)), ev, s=30, color=c, marker="D", zorder=3)
ax.set_ylim(-0.05, 1.3)
base(ax, "what is left of a displacement after 20 steps", "Persistence: does any direction hold a value?", "c")
# (d) marginal count
ax = axs[1, 1]
for tag, c in (("A", cA), ("B", cB)):
    for f in FP[tag]["fits"]:
        if f.get("n_points", 0) < 5: continue
        ax.scatter(jit(f, tag), 100 * f["marginal"] / f["N"], s=22, color=c, zorder=3)
ax.axhline(1.0, color=cA, ls="--", lw=1.4); ax.axhline(0.0, color=cB, ls="--", lw=1.4)
ax.axhline(100 * 33 / 128, color=F.INK3, ls=":", lw=1.4)
ax.text(1.1, 100 * 33 / 128 + 0.8, "SURF model: 33 of 128", color=F.INK2, fontsize=7.5, ha="right")
ax.text(-4.3, 2.0, "truth A: 1 of 100", color=cA, fontsize=7.5); ax.text(-4.3, -2.6, "truth B: none", color=cB, fontsize=7.5)
ax.set_ylim(-4, 34)
base(ax, "marginal directions at fixed points (% of units)", "Fixed-point analysis, done the SURF way", "d", event=False)
h = [plt.Line2D([], [], marker="o", ls="", color=cA, label="free students of teacher A (internal attractor)"),
     plt.Line2D([], [], marker="o", ls="", color=cB, label="free students of teacher B (plan held by an outside input)"),
     plt.Line2D([], [], marker="D", ls="", color=F.INK2, label="students told when inputs arrive"),
     plt.Line2D([], [], ls="--", color=F.INK3, label="truth, measured on the teacher")]
fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.07))
fig.savefig("figs/figP1_plan.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- figP2
fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.7), gridspec_kw=dict(wspace=0.36, width_ratios=[1.1, 1, 1.25]))
ax = axs[0]
oracle = dict(A=0.398, B=0.347)
for j, (tag, c) in enumerate((("A", cA), ("B", cB))):
    fr = [f["cosmooth"] for f in P[tag]["fits"] if f["lam"] <= 1.0]
    ev = [e["cosmooth"] for e in EV[tag]["event"]]
    ax.scatter(j * 2.4 + np.linspace(-0.25, 0.25, len(fr)), fr, s=20, color=c, zorder=3)
    ax.scatter(j * 2.4 + 1 + np.linspace(-0.1, 0.1, len(ev)), ev, s=32, color=c, marker="D", zorder=3)
    ax.plot([j * 2.4 - 0.45, j * 2.4 + 1.45], [oracle[tag]] * 2, color=F.INK3, ls=":", lw=1.4)
ax.text(1.45, oracle["A"] + 0.006, "true rates", color=F.INK2, fontsize=7.5, ha="right")
ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(["free", "told\nwhen", "free", "told\nwhen"])
ax.text(0.5, -0.24, "teacher A", transform=ax.get_xaxis_transform(), ha="center", color=cA, fontsize=8.5)
ax.text(2.9, -0.24, "teacher B", transform=ax.get_xaxis_transform(), ha="center", color=cB, fontsize=8.5)
ax.set_ylabel("held-out neuron prediction (bits/spike)"); ax.set_ylim(0.05, 0.42)
ax.set_title("Held-out prediction:\nfree and told-when students", fontsize=9); F.panel_label(ax, "a")
ax = axs[1]
for j, (tag, c) in enumerate((("A", cA), ("B", cB))):
    ev = EV[tag]["event"]
    ax.scatter(j + np.linspace(-0.1, 0.1, len(ev)), [abs(e["r_amp_v"]) for e in ev], s=32, color=c, marker="D", zorder=3)
ax.set_xticks([0, 1]); ax.set_xticklabels(["teacher A", "teacher B"]); ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.05, 1.08)
ax.set_ylabel("pull vs plan in the model (|r|)")
ax.set_title("Told-when students: is the plan\ncarried all the way to the pull?", fontsize=9); F.panel_label(ax, "b")
ax = axs[2]
groups = [("k = 1", lambda r: r["k"] == 1), ("k = 4\npenalty 0.1", lambda r: r["k"] == 4 and r["lam"] == 0.1),
          ("k = 4\npenalty 0.0001", lambda r: r["k"] == 4 and r["lam"] < 0.01), ("k = 16", lambda r: r["k"] == 16)]
for j, (lab, fn) in enumerate(groups):
    rr = [r for r in MC if fn(r)]
    xs = j + np.linspace(-0.12, 0.12, len(rr))
    ax.scatter(xs - 0.06, [np.clip(r["swap_all"]["x"][0], -0.1, 1.1) for r in rr], s=26, color=F.OPT, zorder=3)
    ax.scatter(xs + 0.06, [np.clip(r["swap_all"]["y"][0], -0.1, 1.1) for r in rr], s=26, color=F.OPT, marker="s", facecolors="none", zorder=3)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(-0.12, 1.12)
ax.set_ylabel("share of the reach set by the trial's own state")
ax.set_title("Monkey data: where the model keeps the reach\ndepends on its settings", fontsize=9); F.panel_label(ax, "c")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=F.OPT, label="reach x"),
                   plt.Line2D([], [], marker="s", ls="", color=F.OPT, markerfacecolor="none", label="reach y")], loc="upper right", fontsize=7.5)
fig.savefig("figs/figP2_event_mc.png", bbox_inches="tight"); plt.close(fig)
print("ok")
