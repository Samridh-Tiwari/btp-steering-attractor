"""Figure 9: channel count with three random starts each (penalty 0.1)."""
import glob, json, re
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F
F.apply()
rows = []
for p in glob.glob("runs/student_k*_lam0.1_s[012]_steer.json"):
    m = re.search(r"student_k(\d+)_lam0.1_s(\d)_steer", p)
    r = json.load(open(p)); d = r["by_ridge"]["0.01"]["delay"]["fit"]
    if r["eval"]["bps_heldin"] < 0:          # diverged fit
        print("skip diverged", p); continue
    rows.append(dict(k=int(m.group(1)), seed=int(m.group(2)), own=d["frac_full"]["median"],
                     best=d["frac_full_opt"]["median"], cos=r["eval"]["bps_cosmooth"],
                     bps=r["eval"]["bps_heldin"], auto=r["eval"]["autonomy"]))
ks = sorted({r["k"] for r in rows})
fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw=dict(wspace=0.32))
ax = axs[0]
for key, c, lab in (("best", F.OPT, "best k channels"), ("own", F.TEST, "its own k channels")):
    for r in rows:
        ax.scatter(np.log2(r["k"]) + (r["seed"] - 1) * 0.08, r[key] * 100, s=16, color=c, alpha=0.6, lw=0)
    ax.plot(np.log2(ks), [np.mean([r[key] for r in rows if r["k"] == k]) * 100 for k in ks], color=c, lw=2, label=lab)
ax.plot(np.log2(ks), [k / 64 * 100 for k in ks], color=F.RAND, lw=2, label="random k channels")
ax.set_xticks(np.log2(ks)); ax.set_xticklabels(ks); ax.set_xlabel("number of inferred channels k")
ax.set_ylabel("fraction of full authority (%)"); ax.set_ylim(0, 105)
ax.set_title("Aim against channel count (dots: seeds)", fontsize=9); ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
F.panel_label(ax, "a")
ax = axs[1]
for r in rows:
    ax.scatter(np.log2(r["k"]) + (r["seed"] - 1) * 0.08, r["cos"], s=16, color=F.OPT, alpha=0.6, lw=0)
ax.plot(np.log2(ks), [np.mean([r["cos"] for r in rows if r["k"] == k]) for k in ks], color=F.OPT, lw=2)
ax.set_xticks(np.log2(ks)); ax.set_xticklabels(ks); ax.set_xlabel("number of inferred channels k")
ax.set_ylabel("held-out neuron prediction (bits/spike)")
ax.set_title("Generalisation against channel count", fontsize=9); F.panel_label(ax, "b")
fig.savefig("figs/fig9_channel_count.png"); plt.close(fig)
for k in ks:
    rr = [r for r in rows if r["k"] == k]
    print(k, len(rr), "own", np.round([r["own"] for r in rr], 3), "best", np.round([r["best"] for r in rr], 3),
          "cos", np.round([r["cos"] for r in rr], 3), "auto", np.round([r["auto"] for r in rr], 2))
