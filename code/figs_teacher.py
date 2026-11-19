"""Figures 1-3: the teacher (ground truth), the decoder pitfall, causal steering."""
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as F
from teacher import load_teacher, cued_response, run_noisy, pullshape
import control as C
from teacher_steer import slow_eigs

F.apply()
torch.set_num_threads(1)
TPATH = "runs/teacher_g0.9_n0.05_s0.pt"
net, _ = load_teacher(TPATH)
W = net.W_rec.detach().numpy(); bvec = net.b.detach().numpy()


# ---------------------------------------------------------------- Figure 1
def fig1():
    fig = plt.figure(figsize=(11.0, 3.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[1.25, 1.05, 1],
                          hspace=0.08, wspace=0.55)
    ax = fig.add_subplot(gs[0, 0]); axi = fig.add_subplot(gs[1, 0], sharex=ax)
    T = 150; tl, tg = 14, 80
    vs = [0.3, 0.6, 0.95]
    u = np.zeros((3, T, 2), np.float32)
    for i, v in enumerate(vs):
        u[i, tl:tl + 5, 0] = v; u[i, tg:tg + 3, 1] = 1.0
    with torch.no_grad():
        y, _ = run_noisy(net, torch.from_numpy(u), 0.0)
    t = np.arange(T) * 0.02 - tg * 0.02
    shades = [F.SEQ[2], F.SEQ[4], F.SEQ[6]]
    for i, v in enumerate(vs):
        tgt = np.zeros(T); tgt[tg:] = v * pullshape(T - tg)
        ax.plot(t, tgt, color=F.GRID, lw=5, solid_capstyle="round", zorder=1)
        ax.plot(t, y[i, :, 0].numpy(), color=shades[i], lw=1.8, zorder=2)
        ax.text(t[tg + 38] + 0.05, v, f"v = {v}", color=F.INK2, fontsize=7.5, va="center")
    ax.set_ylabel("lever output"); ax.set_ylim(-0.08, 1.15)
    ax.set_title("Pull size follows the held plan"); F.panel_label(ax, "a")
    plt.setp(ax.get_xticklabels(), visible=False)
    axi.fill_between(t, 0, u[2, :, 0], color=F.TEST, step="mid", lw=0, alpha=0.9)
    axi.fill_between(t, 0, u[2, :, 1], color=F.GO, step="mid", lw=0, alpha=0.9)
    axi.text(t[tl] + 0.12, 0.55, "value", color=F.INK2, fontsize=7.5)
    axi.text(t[tg] + 0.1, 0.55, "go cue", color=F.INK2, fontsize=7.5)
    axi.set_yticks([]); axi.set_ylim(0, 1.15); axi.grid(False)
    axi.set_xlabel("time from go cue (s)")
    ax.legend(handles=[Line2D([], [], color=F.GRID, lw=5, label="target"),
                       Line2D([], [], color=F.SEQ[4], lw=1.8, label="teacher")],
              loc="upper left", fontsize=7.5)

    # b: the approximate line attractor — continuous on task timescales,
    #    draining to a few fixed points over a minute of silence
    rng = np.random.default_rng(5)
    with torch.no_grad():
        uu, z, m, meta = cued_response(400, rng=rng, return_meta=True)
        _, H = run_noisy(net, uu, 0.0)
        He = torch.stack([H[b_, meta["t_go"][b_] - 1] for b_ in range(400)])
        h = He.clone()
        for _ in range(3000):
            h = net.f_autonomous(h)
    He = He.numpy(); Hs = h.numpy()
    mu = He.mean(0); Hc = He - mu
    U_, S_, Vt = np.linalg.svd(Hc, full_matrices=False)
    var = S_ ** 2 / (S_ ** 2).sum()
    pc = Hc @ Vt[:2].T; pcs = (Hs - mu) @ Vt[:2].T
    ax = fig.add_subplot(gs[:, 1])
    cmap = plt.matplotlib.colors.ListedColormap(F.SEQ[1:])
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=meta["v"], cmap=cmap, s=10, lw=0, vmin=0.2, vmax=1.0,
                    label="end of delay")
    ax.scatter(pcs[:, 0], pcs[:, 1], s=60, facecolor="none", edgecolor=F.INK2, lw=1.0,
               label="after 60 s of silence")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03); cb.set_label("plan v", color=F.INK2, labelpad=2)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
    ax.legend(loc="lower left", fontsize=7, handletextpad=0.2, borderaxespad=0.2)
    ax.set_title("A line, slowly draining to points"); F.panel_label(ax, "b")

    # c: spectrum at a real end-of-delay state
    ax = fig.add_subplot(gs[:, 2])
    hmid = He[np.argsort(meta["v"])[200]]
    f = np.tanh(W @ hmid + bvec); A = (1 - f ** 2)[:, None] * W
    ev = np.linalg.eigvals(A)
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color=F.INK3, lw=0.8, ls="--")
    o = np.argsort(-np.abs(ev))
    ax.scatter(ev[o[1:]].real, ev[o[1:]].imag, s=9, color=F.RAND, lw=0)
    ax.scatter([ev[o[0]].real], [ev[o[0]].imag], s=40, color=F.OPT, lw=0, zorder=3)
    ax.annotate(f"slow mode |λ| = {abs(ev[o[0]]):.3f}", (ev[o[0]].real, 0),
                xytext=(0.05, 0.62), textcoords="data", fontsize=7.5, color=F.INK2,
                arrowprops=dict(arrowstyle="-", color=F.INK3, lw=0.7))
    ax.set_aspect("equal"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Re λ"); ax.set_ylabel("Im λ")
    ax.set_title(f"One slow mode; next |λ| = {abs(ev[o[1]]):.2f}"); F.panel_label(ax, "c")
    fig.savefig("figs/fig1_teacher.png"); plt.close(fig)
    return dict(pc_var=var[:3].tolist(), lam1=float(abs(ev[o[0]])), lam2=float(abs(ev[o[1]])))


# ---------------------------------------------------------------- Figure 2
def fig2():
    rng = np.random.default_rng(11)
    with torch.no_grad():
        u, z, m, meta = cued_response(400, rng=rng, return_meta=True)
        _, H = run_noisy(net, u, 0.0)
    H = H.numpy(); v, tl, tg = meta["v"], meta["t_load"], meta["t_go"]
    X = np.array([H[b, t] for b in range(400) for t in range(tl[b] + 15, tg[b])])
    y = np.array([v[b] for b in range(400) for t in range(tl[b] + 15, tg[b])])
    Xc = X - X.mean(0); yc = y - y.mean()
    hop = H[200, (tl[200] + 5 + tg[200]) // 2]
    f = np.tanh(W @ hop + bvec); A = (1 - f ** 2)[:, None] * W
    Ps, _ = C.slow_subspace(A, 1)
    ridges = np.logspace(-7, 2, 19)
    r2s, es = [], []
    for lam in ridges:
        w = np.linalg.solve(Xc.T @ Xc + lam * len(X) * np.eye(W.shape[0]), Xc.T @ yc)
        r2s.append(1 - ((Xc @ w - yc) ** 2).sum() / (yc ** 2).sum())
        es.append(float(np.sum((Ps.T @ w) ** 2) / np.sum(w ** 2)))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.semilogx(ridges, r2s, color=F.OPT, marker="o", ms=4, label="decoding accuracy (R²)")
    ax.semilogx(ridges, es, color=F.TEST, marker="o", ms=4,
                label="share of decoder axis in the slow subspace")
    ax.set_ylim(-0.03, 1.05); ax.set_xlabel("decoder ridge penalty")
    ax.set_ylabel("fraction")
    ax.set_title("Same decoding accuracy, any geometry you like")
    ax.legend(loc="lower right")
    i1 = int(np.argmin(np.abs(ridges - 1e-6))); i2 = int(np.argmin(np.abs(ridges - 1.0)))
    for i in (i1, i2):
        ax.annotate(f"{es[i]*100:.0f}%", (ridges[i], es[i]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=7.5, color=F.INK2)
    fig.savefig("figs/fig2_decoder_pitfall.png"); plt.close(fig)
    return dict(ridges=ridges.tolist(), r2=list(map(float, r2s)), e_slow=es)


# ---------------------------------------------------------------- Figure 3
def fig3():
    r = json.load(open("runs/teacher_steer.json"))
    fig, axs = plt.subplots(1, 4, figsize=(13.6, 3.25),
                            gridspec_kw=dict(width_ratios=[1.35, 1, 1, 0.85], wspace=0.55))
    # a: authority dot plot
    ax = axs[0]
    rows = [("optimal, 1 ch", F.OPT, None), ("value channel", F.TEST, "value"),
            ("go channel", F.GO, "go"), ("random, 1 ch", F.RAND, None)]
    for j, (lab, col, key) in enumerate(rows):
        yy = len(rows) - 1 - j
        for w, filled, dy in (("load", False, 0.13), ("delay", True, -0.13)):
            if key is not None:
                d = r[w][key]["frac_full"]
                x, lo, hi = d["median"], d["q25"], d["q75"]
            elif lab.startswith("optimal"):
                x = lo = hi = r[w]["opt_curve"]["median"][0]
            else:
                x = lo = hi = 1 / 100
            ax.plot([lo, hi], [yy + dy] * 2, color=col, lw=1.6, solid_capstyle="round")
            ax.scatter([x], [yy + dy], s=46, color=col if filled else F.SURF,
                       edgecolor=col, lw=1.6, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(0.006, 1.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r_[0] for r_ in rows][::-1])
    ax.set_xlabel("authority over pull size\n(fraction of stimulating all 100 units)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=F.SURF, mec=F.INK2, label="load window"),
                       Line2D([], [], marker="o", ls="", mfc=F.INK2, mec=F.INK2, label="delay window")],
              loc="lower right", fontsize=7.5, handletextpad=0.2)
    ax.set_title("Positive control passes"); F.panel_label(ax, "a")

    # b: optimal-k curve
    ax = axs[1]
    for w, ls in (("delay", "-"), ("load", "--")):
        c = r[w]["opt_curve"]
        ax.plot(c["k"], c["median"], color=F.OPT, ls=ls, marker="o", ms=4, label=f"optimal, {w}")
    ks = np.array(r["delay"]["opt_curve"]["k"])
    ax.plot(ks, ks / 100, color=F.RAND, marker="o", ms=4, label="random, either")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_ylim(0, 1.05); ax.set_xlabel("number of channels k")
    ax.set_ylabel("fraction of full authority")
    ax.set_title("One aimed channel is enough"); ax.legend(loc="center right")
    F.panel_label(ax, "b")

    # c: linearity
    ax = axs[2]
    for w, ls in (("delay", "-"), ("load", "--")):
        L = r[w]["linearity"]
        eps = [l["eps"] for l in L]
        ax.plot(eps, [l["ratio_median"] for l in L], color=F.INK2, ls=ls, marker="o", ms=4, label=w)
        ax.fill_between(eps, [l["ratio_q25"] for l in L], [l["ratio_q75"] for l in L],
                        color=F.GRID, lw=0)
    ax.axhline(1, color=F.INK3, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0, 1.15)
    ax.set_xlabel("requested change in pull size")
    ax.set_ylabel("achieved / predicted")
    ax.set_title("Linear prediction holds to ~0.1"); ax.legend(loc="lower left")
    F.panel_label(ax, "c")

    # d: geometry — aim at the left eigenvector, not along the manifold
    ax = axs[3]
    g = r["geometry"]
    a = np.radians(g["s_vs_right"])
    ax.annotate("", xy=(1, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=F.INK2, lw=1.8))
    ax.annotate("", xy=(np.cos(a), np.sin(a)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=F.OPT, lw=1.8))
    th = np.linspace(0, a, 40)
    ax.plot(0.32 * np.cos(th), 0.32 * np.sin(th), color=F.INK3, lw=0.8)
    ax.text(0.38 * np.cos(a / 2), 0.38 * np.sin(a / 2), f"{g['s_vs_right']:.0f}°",
            fontsize=8, color=F.INK2, va="center")
    ax.text(1.0, -0.12, "manifold\ntangent", fontsize=7.5, color=F.INK2, ha="right", va="top")
    ax.text(np.cos(a) - 0.02, np.sin(a) + 0.06, "best input\ndirection", fontsize=7.5,
            color=F.OPT, ha="right", va="bottom")
    ax.text(0.0, -0.62, f"best input = left eigenvector\n(to {g['s_vs_Dleft']:.1f}°); left vs right {g['right_vs_left']:.0f}°",
            fontsize=7, color=F.INK2, va="top")
    ax.set_xlim(-0.08, 1.12); ax.set_ylim(-0.95, 0.85); ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Aim isn't 'along the manifold'"); F.panel_label(ax, "d")
    fig.savefig("figs/fig3_teacher_steering.png"); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("figs", exist_ok=True)
    out = dict(fig1=fig1(), fig2=fig2())
    fig3()
    json.dump(out, open("runs/figs_teacher.json", "w"), indent=1)
    print(json.dumps({k: {kk: (vv if not isinstance(vv, list) or len(vv) < 6 else "...")
                          for kk, vv in v.items()} for k, v in out.items()}, indent=1))
