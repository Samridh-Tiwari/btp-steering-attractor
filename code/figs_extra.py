"""Figure 6: behaviour vs whole-state objective. Figure 7: same settings, different random starts."""
import json
import numpy as np
import matplotlib.pyplot as plt

import figstyle as F
from figs_student import load_all, sel, D

F.apply()


def fig6():
    st = json.load(open("runs/state_target.json"))
    panels = [("teacher", "a", "Teacher, its 2 real input channels\n(100 units)", "real_inputs", "real inputs"),
              ("runs/student_k4_lam0.0001_s0.pt", "b", "Student, 4 inferred channels\n(penalty 1e-4, 64 units)", "fit", "inferred"),
              ("runs/student_k4_lam0.1_s0.pt", "c", "Student, 4 inferred channels\n(penalty 0.1, 64 units)", "fit", "inferred")]
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.3), gridspec_kw=dict(wspace=0.35))
    for ax, (key, lab, title, name, nlab) in zip(axs, panels):
        r = st[key]["result"]
        sets = [("random", "random", F.RAND), (name, nlab, F.TEST), ("opt_behaviour", "best-aimed\nfor behaviour", F.OPT)]
        x = np.arange(2); w = 0.26
        for j, (s, l, c) in enumerate(sets):
            vals = [r[s]["behaviour"]["median"] * 100, r[s]["state"]["median"] * 100]
            lo = [r[s]["behaviour"]["q25"] * 100, r[s]["state"]["q25"] * 100]
            hi = [r[s]["behaviour"]["q75"] * 100, r[s]["state"]["q75"] * 100]
            ax.bar(x + (j - 1) * w, vals, w * 0.92, color=c, label=l)
            ax.errorbar(x + (j - 1) * w, vals, yerr=[np.subtract(vals, lo), np.subtract(hi, vals)],
                        fmt="none", ecolor=F.INK, lw=0.8, capsize=2)
        ax.axhline(90, color=F.INK, lw=0.8, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(["move the\nlever trace", "move the\nwhole state"])
        ax.set_ylim(0, 100); ax.set_ylabel("error removed (%)")
        ax.set_title(title, fontsize=9); F.panel_label(ax, lab)
    axs[0].text(1.42, 91.5, "every unit actuated", fontsize=7, ha="right", color=F.INK)
    axs[0].legend(loc="upper center", fontsize=7, frameon=False, ncol=1, bbox_to_anchor=(0.32, 0.86))
    fig.savefig("figs/fig6_objective.png"); plt.close(fig)


def fig7(rows):
    cfgs = [(4, 1e-4), (4, 0.1)]
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.1), gridspec_kw=dict(wspace=0.4))
    metrics = [("aim of inferred channels\n(% of full actuation)", lambda r: D(r)["frac_full"]["median"] * 100),
               ("autonomy (SURF criterion)", lambda r: r["eval"]["autonomy"]),
               ("held-out neuron prediction\n(bits / spike)", lambda r: r["eval"]["bps_cosmooth"])]
    for ax, (lab, f), pl in zip(axs, metrics, "abc"):
        for i, (k, lam) in enumerate(cfgs):
            rr = sel(rows, k=k, lam=lam, teacher=1)
            v = [f(r) for r in rr]
            jit = np.linspace(-0.08, 0.08, len(v)) if len(v) > 1 else [0]
            ax.scatter(i + np.array(jit), v, s=28, color=F.TEST, zorder=3)
            for xx, vv, r in zip(i + np.array(jit), v, rr):
                ax.annotate(f"seed {r['seed']}", (xx, vv), xytext=(6, 0), textcoords="offset points",
                            fontsize=6.5, va="center", color=F.INK)
        ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([f"penalty {l:g}" for _, l in cfgs])
        ax.set_xlim(-0.5, len(cfgs) - 0.5); ax.set_ylabel(lab); F.panel_label(ax, pl)
    axs[0].set_title("Same data, same settings,\ndifferent random start", fontsize=9)
    fig.savefig("figs/fig7_seeds.png"); plt.close(fig)


if __name__ == "__main__":
    import os
    fig6() if os.path.exists("runs/state_target.json") else None
    fig7(load_all()) if len(__import__("glob").glob("runs/student_k4_lam0.1_s*_steer.json")) > 3 else None
    fig8()
    print("ok")


def fig8():
    r = json.load(open("runs/rho_sweep.json"))
    mult = np.array(r["mult"])
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.4), gridspec_kw=dict(wspace=0.3), sharey=True)
    sets = [("random_4", "random 4", F.RAND), ("real_inputs", "real inputs (2)", F.TEST),
            ("best_4_behaviour", "best 4 for behaviour", F.OPT)]
    for ax, goal, title, lab in ((axs[0], "behaviour", "Goal: move the lever trace", "a"),
                                 (axs[1], "state", "Goal: move the whole state", "b")):
        for key, l, c in sets:
            ax.loglog(mult, np.clip(np.array(r[key][goal]) * 100, 1e-4, None), color=c, lw=2, label=l)
        ax.axhline(0.1, color=F.INK, lw=0.9, ls=":")
        ax.axvline(1, color=F.INK3, lw=0.8, ls="--")
        ax.set_xlabel("effort cost, relative to the fair setting")
        ax.set_title(title, fontsize=9); F.panel_label(ax, lab)
        ax.set_ylim(1e-3, 150)
    axs[0].set_ylabel("error removed (%)")
    axs[0].text(1.3, 3e-3, "fair setting:\nall units remove 90%", fontsize=7, color=F.INK)
    axs[1].text(0.012, 0.13, "SURF: 0.1%", fontsize=7, color=F.INK)
    axs[0].legend(loc="lower left", fontsize=7, frameon=False)
    fig.savefig("figs/fig8_effort_cost.png"); plt.close(fig)
