"""Figures 4-5: the fitted students — the cfg01/cfg06 trade-off with ground truth, and steering."""
import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as F

F.apply()
RIDGE = "0.01"
N_STUDENT = 64


def load_all():
    rows = []
    for p in sorted(glob.glob("runs/student_k*_lam*_s*_steer.json")):
        r = json.load(open(p))
        m = re.search(r"student_k(\d+)_lam([0-9.e-]+)_s(\d+)(_t2)?_steer", p)
        ev = json.load(open(p.replace("_steer.json", "_eval.json")))
        r.update(k=int(m.group(1)), lam=float(m.group(2)), seed=int(m.group(3)),
                 teacher=2 if m.group(4) else 1, ev_full=ev)
        rows.append(r)
    return rows


def sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def D(r, window="delay"):
    return r["by_ridge"][RIDGE][window]["fit"]


# ---------------------------------------------------------------- Figure 4
def fig4(rows):
    pen = sorted(sel(rows, k=4, seed=0, teacher=1), key=lambda r: r["lam"])
    lam = np.array([r["lam"] for r in pen])
    fig, axs = plt.subplots(1, 4, figsize=(13.4, 3.0), gridspec_kw=dict(wspace=0.5))

    ax = axs[0]
    ax.semilogx(lam, [r["eval"]["bps_heldin"] for r in pen], color=F.INK2, marker="o", ms=4)
    ax.set_ylabel("bits / spike"); ax.set_xlabel("controller penalty")
    ax.set_title("Fit to recorded neurons"); F.panel_label(ax, "a")

    ax = axs[1]
    ax.semilogx(lam, [r["eval"]["bps_cosmooth"] for r in pen], color=F.OPT, marker="o", ms=4)
    ax.set_ylabel("bits / spike"); ax.set_xlabel("controller penalty")
    ax.set_title("Prediction of held-out neurons"); F.panel_label(ax, "b")

    ax = axs[2]
    ax.semilogx(lam, [r["eval"]["autonomy"] for r in pen], color=F.INK2, marker="o", ms=4,
                label="autonomy (SURF criterion)")
    PT = {(f["lam"], f["seed"]): f for f in json.load(open("runs/plan_tests_ABT2.json"))["A"]["fits"]}
    ax.semilogx(lam, [PT[(r["lam"], 0)]["swap_all"]["own"] for r in pen], color=F.OPT,
                marker="o", ms=4, label="share of the pull set by the\nmodel's own state (swap test)")
    ax.axhline(0, color=F.INK3, lw=0.8, ls=":")
    ax.set_ylim(-0.7, 1.05); ax.set_xlabel("controller penalty")
    ax.set_title("Does the model run on its own?"); ax.legend(loc="upper left", fontsize=6.5)
    F.panel_label(ax, "c")

    ax = axs[3]
    t = np.arange(-20, 40) * 0.02
    for r, col, lab in ((pen[0], F.RAND, f"penalty {pen[0]['lam']:g}"),
                        (sel(pen, lam=0.1)[0], F.TEST, "penalty 0.1")):
        u = np.array(r["ev_full"]["u_go_locked"])
        ax.plot(t, u / np.median(u[:15]), color=col, label=lab)
    ax.axvline(0, color=F.INK3, lw=0.8, ls=":")
    ax.set_xlabel("time from go cue (s)"); ax.set_ylabel("controller |u| / baseline")
    ax.set_title("Controller becomes a cue signal"); ax.legend(loc="upper right", fontsize=7)
    F.panel_label(ax, "d")
    fig.savefig("figs/fig4_students_tradeoff.png"); plt.close(fig)


# ---------------------------------------------------------------- Figure 5
def fig5(rows, teacher):
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 7.2), gridspec_kw=dict(wspace=0.55, hspace=0.5))
    axs = axs.ravel()
    # a: authority vs penalty
    pen = sorted(sel(rows, k=4, seed=0, teacher=1), key=lambda r: r["lam"])
    lam = np.array([r["lam"] for r in pen])
    ax = axs[0]
    ax.semilogx(lam, [D(r)["frac_full_opt"]["median"] for r in pen], color=F.OPT, marker="o", ms=4,
                label="best possible 4")
    fit = [D(r)["frac_full"] for r in pen]
    ax.semilogx(lam, [f["median"] for f in fit], color=F.TEST, marker="o", ms=4, label="its own 4 channels")
    ax.fill_between(lam, [f["q25"] for f in fit], [f["q75"] for f in fit], color=F.TEST, alpha=0.15, lw=0)
    ax.semilogx(lam, [4 / N_STUDENT] * len(lam), color=F.RAND, marker="o", ms=4, label="random 4")
    ax.set_ylim(0, 1.05); ax.set_xlabel("controller penalty")
    ax.set_ylabel("fraction of full authority\nover pull size (delay)")
    ax.set_title("More faithful, worse aimed"); ax.legend(loc="center left", fontsize=7)
    F.panel_label(ax, "a")

    # b: authority vs number of channels, at the faithful penalty
    ax = axs[1]
    for lamv, ls, lab in ((0.1, "-", "penalty 0.1"), (0.01, "--", "penalty 0.01")):
        ks_rows = sorted(sel(rows, lam=lamv, seed=0, teacher=1), key=lambda r: r["k"])
        if len(ks_rows) < 2:
            continue
        ks = np.array([r["k"] for r in ks_rows])
        ax.plot(ks, [D(r)["frac_full"]["median"] for r in ks_rows], color=F.TEST, ls=ls,
                marker="o", ms=4, label=f"own channels, {lab}")
        ax.plot(ks, [D(r)["frac_full_opt"]["median"] for r in ks_rows], color=F.OPT, ls=ls,
                marker="o", ms=4, label=f"best possible, {lab}")
    kk = np.array([1, 2, 4, 8, 16])
    ax.plot(kk, kk / N_STUDENT, color=F.RAND, marker="o", ms=4, label="random")
    ax.set_xscale("log", base=2); ax.set_xticks(kk); ax.set_xticklabels(kk)
    ax.set_ylim(0, 1.05); ax.set_xlabel("number of inferred channels k")
    ax.set_ylabel("fraction of full authority")
    ax.set_title("More channels don't fix aim"); ax.legend(loc="center right", fontsize=6.5)
    F.panel_label(ax, "b")

    # c, d: every channel set on one fair scale — its random floor, its optimal
    # ceiling, and where it actually lands. ×random alone is unfair because its
    # ceiling depends on k and N (100× for 1 of 100, 16× for 4 of 64).
    for ax, window, title, lab_ in ((axs[2], "delay", "During the hold", "c"),
                                    (axs[3], "load", "While the plan loads", "d")):
        items = []
        for name, key, col, k, N in (("teacher value ch. (1)", "value", F.TEST, 1, 100),
                                     ("teacher go ch. (1)", "go", F.GO, 1, 100),
                                     ("teacher both (2)", "both", F.INK2, 2, 100)):
            d = teacher[window][key]
            items.append((name, col, [d["frac_full"]["median"]], k / N,
                          teacher[window]["opt_curve"]["median"][k - 1]))
        for lamv, name in ((1e-4, "student 1e-4 (4)"), (1e-2, "student 1e-2 (4)"),
                           (0.1, "student 0.1 (4)")):
            rr = sel(rows, k=4, lam=lamv)
            if rr:
                items.append((name, F.TEST,
                              [D(r, window)["frac_full"]["median"] for r in rr], 4 / N_STUDENT,
                              float(np.median([D(r, window)["frac_full_opt"]["median"] for r in rr]))))
        for j, (name, col, vals, lo, hi) in enumerate(items):
            yy = len(items) - 1 - j
            ax.plot([lo, hi], [yy, yy], color=F.GRID, lw=5, solid_capstyle="round", zorder=1)
            ax.scatter([lo], [yy], marker="|", s=90, color=F.RAND, lw=2, zorder=2)
            ax.scatter([hi], [yy], marker="|", s=90, color=F.OPT, lw=2, zorder=2)
            mk = "s" if name.startswith("student") else "o"
            ax.scatter(vals, [yy] * len(vals), color=col, s=36, marker=mk, zorder=3, lw=0)
        ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in items][::-1], fontsize=7.5)
        ax.set_xscale("log"); ax.set_xlim(0.007, 1.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("fraction of full authority")
        ax.set_title(title); F.panel_label(ax, lab_)
    axs[3].legend(handles=[Line2D([], [], marker="|", ls="", ms=9, mew=2, color=F.RAND, label="random, same k"),
                           Line2D([], [], marker="|", ls="", ms=9, mew=2, color=F.OPT, label="optimal, same k"),
                           Line2D([], [], marker="o", ls="", color=F.TEST, label="real input channel"),
                           Line2D([], [], marker="s", ls="", color=F.TEST, label="inferred channels")],
                  loc="lower right", fontsize=6.5, handletextpad=0.3, borderaxespad=0.2)
    fig.savefig("figs/fig5_students_steering.png"); plt.close(fig)


if __name__ == "__main__":
    rows = load_all()
    teacher = json.load(open("runs/teacher_steer.json"))
    print(f"{len(rows)} student results:",
          sorted({(r['k'], r['lam'], r['seed'], r['teacher']) for r in rows}))
    fig4(rows)
    fig5(rows, teacher)
