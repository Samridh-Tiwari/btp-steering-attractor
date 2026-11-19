"""Figures for the full-LFADS section (fig13, fig14), in the report's figure style."""
import glob
import json
import re

import numpy as np
import matplotlib.pyplot as plt

import figstyle as F

F.apply()
C_STD = "#2a78d6"      # LFADS, standard (IC encoder sees the whole trial)
C_LITE = "#eb6834"     # simplified free student
C_TW = "#1baf7a"       # told-when student
C_CAU = "#eda100"      # LFADS, causal IC
C_NS = "#e87ba4"       # LFADS, causal IC, controller not sampled in training
C_T = F.INK            # teacher (truth)
C_R = F.RAND

LT = json.load(open("runs/lfads_tests.json"))
RQ = json.load(open("runs/rate_quality.json"))
VT = json.load(open("runs/v_timecourse.json"))
GC = json.load(open("runs/gate_check.json"))
TS = json.load(open("runs/teacher_steer.json"))
PT = json.load(open("runs/plan_tests_ABT2.json"))
try:
    MCF = json.load(open("runs/lfads_mc_tests.json"))
except Exception:
    MCF = {}


def group(tag, kind):
    """kind: std | cau | ns (causal, not sampled) | stdns"""
    pat = {"std": rf"runs/lfads_{tag}_kl[0-9.e-]+_s\d_tr192\.pt$",
           "cau": rf"runs/lfads_{tag}c_kl[0-9.e-]+_s\d_tr192\.pt$",
           "ns": rf"runs/lfads_{tag}c_kl[0-9.e-]+_s\d_tr192_nosamp\.pt$",
           "stdns": rf"runs/lfads_{tag}_kl[0-9.e-]+_s\d_tr192_nosamp\.pt$"}[kind]
    return {p: r for p, r in LT.items() if re.match(pat, p)}


def jitter(n, w=0.12, seed=0):
    return np.random.default_rng(seed).uniform(-w, w, n) if n > 1 else np.zeros(n)


def strip(ax, x, vals, color, marker="o", label=None, seed=0, size=34):
    vals = np.asarray(vals, float)
    ax.scatter(x + jitter(len(vals), seed=seed), vals, s=size, color=color, marker=marker, zorder=3,
               edgecolor="white", linewidth=0.8, label=label)


def lite_auth(tag="A"):
    suf = "" if tag == "A" else "_" + tag
    out = []
    for p in sorted(glob.glob(f"runs/student_k4_lam*_s?{suf}_steer.json")):
        lam = float(re.search(r"lam([0-9.e-]+)_s", p).group(1))
        if lam > 1:
            continue
        r = json.load(open(p))
        out.append(r["by_ridge"]["0.01"]["delay"]["fit"]["frac_full"]["median"])
    return out


def fig13():
    fig, axs = plt.subplots(2, 3, figsize=(13.2, 8.2))
    fig.subplots_adjust(wspace=0.36, hspace=0.62)
    # (a) single-trial rate accuracy
    ax = axs[0, 0]
    lite = [r["rate_r2"] for r in RQ["A"] if r["kind"] == "free" and r["lam"] <= 1]
    tw = [r["rate_r2"] for r in RQ["A"] if r["kind"] == "toldwhen"]
    cats = [("Simplified", lite, C_LITE, "o"), ("Told-\nwhen", tw, C_TW, "D"),
            ("LFADS", [r["rate_r2"] for r in group("A", "std").values()], C_STD, "s"),
            ("LFADS\ncausal", [r["rate_r2"] for r in group("A", "cau").values()], C_CAU, "^"),
            ("causal,\nnot\nsampled", [r["rate_r2"] for r in group("A", "ns").values()], C_NS, "v")]
    for i, (nm, vals, c, mk) in enumerate(cats):
        strip(ax, i, vals, c, mk, seed=i)
        if len(vals):
            ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, color=F.INK2, lw=1.2, zorder=2)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([c[0] for c in cats], fontsize=7.2)
    ax.set_ylabel("$R^2$ of single-trial rates vs truth"); ax.set_ylim(-0.4, 1.05)
    ax.axhline(0, color=F.INK3, lw=0.8)
    ax.set_title("Single-trial rates (held-out trials for LFADS)")
    F.panel_label(ax, "a")
    # (b) when is the plan known
    ax = axs[0, 1]
    lags = np.array(VT["lags"])
    ax.axvspan(0, 5, color=F.GRID, zorder=0)
    ax.text(2.5, 1.1, "plan cue", ha="center", fontsize=7.5, color=F.INK2)
    ax.plot(lags, VT["teacher"], color=C_T, lw=2, label="teacher")
    ax.plot(lags, np.median(VT["lite"], 0), color=C_LITE, lw=2, label="simplified student")
    def get(p, k):
        return VT.get(p, {}).get(k)
    std = get("runs/lfads_A_kl0.0001_s0_tr192.pt", "test")
    cau_tr = get("runs/lfads_Ac_kl0.0001_s0_tr192.pt", "train"); cau_te = get("runs/lfads_Ac_kl0.0001_s0_tr192.pt", "test")
    ns = get("runs/lfads_Ac_kl0.0001_s0_tr192_nosamp.pt", "test")
    if std: ax.plot(lags, std, color=C_STD, lw=2, label="LFADS (held-out trials)")
    if cau_tr: ax.plot(lags, cau_tr, color=C_CAU, lw=2, ls="--", label="LFADS causal IC, training trials")
    if cau_te: ax.plot(lags, cau_te, color=C_CAU, lw=2, label="LFADS causal IC, held-out trials")
    if ns: ax.plot(lags, ns, color=C_NS, lw=2, label="causal IC, not sampled, held-out")
    ax.set_xlabel("time from plan cue (steps)"); ax.set_ylabel("$R^2$ of plan value $v$ from model state")
    ax.set_ylim(-0.6, 1.2); ax.set_xlim(lags[0], lags[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.6, handlelength=1.6, columnspacing=1.0)
    ax.set_title("When does the model know the plan?")
    F.panel_label(ax, "b")
    # (c) authority of the inferred channels in the delay
    ax = axs[0, 2]
    ax.set_yscale("log")
    real = TS["delay"]["both"]["frac_full"]
    real = real["median"] if isinstance(real, dict) else real
    items = [("Teacher's\nreal\ninputs", [real], C_T, "*"), ("Simplified", lite_auth("A"), C_LITE, "o"),
             ("LFADS", [r["authority"]["frac_full"] for r in group("A", "std").values()], C_STD, "s"),
             ("LFADS\ncausal", [r["authority"]["frac_full"] for r in group("A", "cau").values()], C_CAU, "^"),
             ("causal,\nnot\nsampled", [r["authority"]["frac_full"] for r in group("A", "ns").values()], C_NS, "v")]
    for i, (nm, vals, c, mk) in enumerate(items):
        strip(ax, i, np.array(vals) * 100, c, mk, seed=10 + i, size=60 if mk == "*" else 34)
    ax.axhline(100 * 4 / 64, color=C_R, lw=1.2, ls="--"); ax.text(len(items) - 0.5, 100 * 4 / 64 * 1.15, "random 4", color=F.INK2, fontsize=7.5, ha="right")
    best1 = [r["authority"]["frac_opt1"] for r in group("A", "std").values()]
    if best1:
        ax.axhline(100 * np.median(best1), color=F.OPT, lw=1.2, ls=":")
        ax.text(len(items) - 0.5, 100 * np.median(best1) * 0.72, "best single direction", color=F.INK2, fontsize=7.5, ha="right")
    ax.set_xticks(range(len(items))); ax.set_xticklabels([it[0] for it in items], fontsize=7.2)
    ax.set_ylabel("% of full authority (delay)"); ax.set_ylim(0.01, 200)
    ax.set_title("Inferred channels, teacher A (4 channels)")
    F.panel_label(ax, "c")
    # (d) gate shares
    ax = axs[1, 0]
    rows = [(p, r) for p, r in GC.items()]
    sens = [r["sens_share_on_z95"] for _, r in rows]; push = [r["push_share_on_z95"] for _, r in rows]
    frac = [r["frac_units_z95"] for _, r in rows]
    xs = np.arange(3)
    for x, vals, c, nm in ((0, frac, F.INK3, "units"), (1, sens, F.OPT, "sensitivity"), (2, push, F.TEST, "controller push")):
        ax.bar(x, 100 * np.median(vals), width=0.6, color=c, zorder=2)
        strip(ax, x, 100 * np.array(vals), F.INK, "o", seed=20 + x, size=14)
    ax.set_xticks(xs); ax.set_xticklabels(["units with\nclosed gate", "behavioural\nsensitivity", "controller's\npush"], fontsize=7.5)
    ax.set_ylabel("% on units with update gate $z>0.95$"); ax.set_ylim(0, 105)
    ax.set_title("The gate that holds the memory shuts the input out")
    F.panel_label(ax, "d")
    # (e) open vs closed loop shift gain
    ax = axs[1, 1]
    tA = PT["A"]["teacher"]
    liteA = [f for f in PT["A"]["fits"] if f.get("lam", 1) <= 1]
    def gains(gr, key):
        return [r[key] for r in gr.values()]
    cats = [("Teacher", [tA["shift_gain"]], None, C_T, "*"),
            ("Simplified", [f["shift_gain"] for f in liteA], None, C_LITE, "o"),
            ("LFADS", gains(group("A", "std"), "shift_gain"), gains(group("A", "std"), "shift_gain_closed"), C_STD, "s"),
            ("LFADS\ncausal", gains(group("A", "cau"), "shift_gain"), gains(group("A", "cau"), "shift_gain_closed"), C_CAU, "^"),
            ("causal,\nnot\nsampled", gains(group("A", "ns"), "shift_gain"), gains(group("A", "ns"), "shift_gain_closed"), C_NS, "v")]
    for i, (nm, op, cl, c, mk) in enumerate(cats):
        strip(ax, i - (0.17 if cl is not None else 0), op, c, mk, seed=30 + i, size=60 if mk == "*" else 34)
        if cl is not None and len(cl):
            vals = np.asarray(cl)
            ax.scatter(i + 0.17 + jitter(len(vals), 0.06, 40 + i), vals, s=34, facecolor="white", edgecolor=c,
                       marker=mk, linewidth=1.4, zorder=3)
    ax.scatter([], [], s=34, color=F.INK3, marker="o", label="open loop (inputs kept)")
    ax.scatter([], [], s=34, facecolor="white", edgecolor=F.INK3, marker="o", linewidth=1.4, label="closed loop (controller sees spikes)")
    ax.legend(loc="lower left", fontsize=6.8)
    ax.axhline(0, color=F.INK3, lw=0.8)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([c[0] for c in cats], fontsize=7.2)
    ax.set_ylabel("shift gain"); ax.set_ylim(-0.25, 1.15)
    ax.set_title("Does a push in the delay reach the pull?")
    F.panel_label(ax, "e")
    # (f) kl sweep: autonomy
    ax = axs[1, 2]
    for kind, c, mk, nm in (("std", C_STD, "s", "LFADS"), ("cau", C_CAU, "^", "LFADS causal IC")):
        gr = group("A", kind)
        if not gr:
            continue
        kl = np.array([r["kl_co"] for r in gr.values()]); au = np.array([r["autonomy"] for r in gr.values()])
        ax.scatter(kl * np.exp(jitter(len(kl), 0.08, 50)), au, s=34, color=c, marker=mk, edgecolor="white", linewidth=0.8, zorder=3, label=nm)
        ks = np.unique(kl)
        ax.plot(ks, [np.median(au[kl == k]) for k in ks], color=c, lw=1.4, zorder=2)
    ax.set_xscale("log")
    for k, nm in ((1e-5, "SURF cfg01"), (1e-4, "SURF cfg06")):
        ax.axvline(k, color=F.INK3, lw=0.8, ls=":")
        ax.text(k * 1.1, -0.95, nm, rotation=90, fontsize=7, color=F.INK2, va="bottom")
    ax.set_xlabel("kl_co_scale"); ax.set_ylabel("autonomy (generator alone, $R^2$)")
    ax.set_ylim(-1, 1.05); ax.legend(loc="lower right", fontsize=7)
    ax.set_title("How much the controller is used")
    F.panel_label(ax, "f")
    fig.savefig("figs/fig13_lfads.png")
    print("fig13 saved")


def fig14():
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 7.8))
    fig.subplots_adjust(wspace=0.32, hspace=0.55)
    axs = axs.ravel()
    # (a) authority against kl, teacher A
    ax = axs[0]; ax.set_yscale("log"); ax.set_xscale("log")
    for kind, c, mk, nm in (("std", C_STD, "s", "LFADS"), ("cau", C_CAU, "^", "LFADS causal IC"), ("ns", C_NS, "v", "causal IC, not sampled")):
        gr = group("A", kind)
        if not gr:
            continue
        kl = np.array([r["kl_co"] for r in gr.values()]); au = 100 * np.array([r["authority"]["frac_full"] for r in gr.values()])
        ax.scatter(kl * np.exp(jitter(len(kl), 0.08, 60)), au, s=34, color=c, marker=mk, edgecolor="white", linewidth=0.8, zorder=3, label=nm)
    ax.axhline(100 * 4 / 64, color=C_R, lw=1.2, ls="--"); ax.text(7e-7, 100 * 4 / 64 * 1.2, "random 4", fontsize=7.5, color=F.INK2)
    la = lite_auth("A")
    ax.axhspan(100 * min(la), 100 * max(la), color=C_LITE, alpha=0.15, zorder=0)
    ax.text(7e-7, 100 * max(la) * 1.1, "simplified students", fontsize=7.5, color=F.INK2)
    for k in (1e-5, 1e-4):
        ax.axvline(k, color=F.INK3, lw=0.8, ls=":")
    ax.text(1.1e-5, 0.012, "SURF", fontsize=7, color=F.INK2); ax.text(1.1e-4, 0.012, "SURF", fontsize=7, color=F.INK2)
    ax.set_xlabel("kl_co_scale"); ax.set_ylabel("% of full authority (delay)"); ax.set_ylim(0.01, 200); ax.set_xlim(5e-7, 2e-2)
    ax.legend(loc="center right", fontsize=6.8); ax.set_title("Inferred channels across the penalty, teacher A")
    F.panel_label(ax, "a")
    # (b) other circuits
    ax = axs[1]; ax.set_yscale("log")
    sets = [("A", "teacher A"), ("B", "teacher B\n(no memory)"), ("T2", "teacher A2"), ("An", "teacher A\nwith noise")]
    for i, (tag, nm) in enumerate(sets):
        for kind, c, mk, off in (("std", C_STD, "s", -0.15), ("cau", C_CAU, "^", 0.15)):
            gr = group(tag, kind)
            vals = 100 * np.array([r["authority"]["frac_full"] for r in gr.values()])
            if len(vals):
                strip(ax, i + off, vals, c, mk, seed=70 + i)
    ax.scatter([], [], s=34, color=C_STD, marker="s", label="LFADS"); ax.scatter([], [], s=34, color=C_CAU, marker="^", label="LFADS causal IC")
    ax.axhline(100 * 4 / 64, color=C_R, lw=1.2, ls="--")
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([s_[1] for s_ in sets], fontsize=7.5)
    ax.set_ylabel("% of full authority (delay)"); ax.set_ylim(0.01, 200); ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Other circuits (held-out trials)")
    F.panel_label(ax, "b")
    # (c) teacher B: the truth against standard LFADS
    ax = axs[2]
    tB = PT["B"]["teacher"]
    std = list(group("B", "std").values())
    keys = [("plan known\nat trial start", None, "v_at_start"), ("swap: own\nshare", tB["swap_all"]["own"], "swap_all"),
            ("shift gain", tB["shift_gain"], "shift_gain"), ("persistence", tB["persist_realpos"], "persist_realpos")]
    for i, (nm, truth, key) in enumerate(keys):
        ax.bar(i - 0.18, 0.0 if truth is None else truth, width=0.34, color=F.INK2, zorder=2, label="teacher B (truth)" if i == 0 else None)
        vals = [(r[key]["own"] if isinstance(r[key], dict) else r[key]) for r in std]
        if vals:
            ax.bar(i + 0.18, np.median(vals), width=0.34, color=C_STD, zorder=2, label="LFADS (held-out trials)" if i == 0 else None)
            strip(ax, i + 0.18, vals, F.INK, "o", seed=90 + i, size=12)
    for i in range(len(keys)):
        ax.text(i - 0.18, 0.03, "0", ha="center", fontsize=8, color=F.INK)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k[0] for k in keys], fontsize=7.5)
    ax.set_ylim(0, 1.7); ax.set_ylabel("value"); ax.legend(loc="upper left", fontsize=7)
    ax.set_title("A circuit with no memory: LFADS reports one")
    F.panel_label(ax, "c")
    # (d) MC_Maze
    ax = axs[3]; ax.set_yscale("log")
    lite_mc, best_mc = [], []
    for p in sorted(glob.glob("runs/mc_student_k4_lam*_s?_steer.json")):
        r = json.load(open(p))["by_ridge"]["0.001"]["delay"]["fit"]
        lite_mc.append(r["frac_full"]["median"]); best_mc.append(r["frac_full_opt"]["median"])
    full = [r["authority"]["frac_full"] for r in MCF.values()]
    full_best = [r["authority"]["frac_opt"] for r in MCF.values()]
    strip(ax, 0, 100 * np.array(lite_mc), C_LITE, "o", seed=80)
    strip(ax, 1, 100 * np.array(full), C_STD, "s", seed=81)
    strip(ax, 0, 100 * np.array(best_mc), F.OPT, "_", seed=82, size=90)
    strip(ax, 1, 100 * np.array(full_best), F.OPT, "_", seed=83, size=90)
    ax.axhline(100 * 4 / 64, color=C_R, lw=1.2, ls="--"); ax.text(1.55, 100 * 4 / 64 * 1.2, "random 4", fontsize=7.5, color=F.INK2, ha="right")
    ax.text(1.55, 130, "best 4 (dashes)", fontsize=7.5, color=F.INK2, ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Simplified", "LFADS"]); ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("% of full authority (last 300 ms of delay)"); ax.set_ylim(0.01, 200)
    ax.set_title("MC_Maze (monkey motor cortex): inferred channels")
    F.panel_label(ax, "d")
    fig.savefig("figs/fig14_lfads_robust.png")
    print("fig14 saved")


if __name__ == "__main__":
    fig13(); fig14()
