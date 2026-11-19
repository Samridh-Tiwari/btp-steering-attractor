"""SURF-style fixed-point analysis on the students: find fixed (or slow) points of the
autonomous generator (controller off) starting from delay-period states, by minimising
q = 0.5 |f(g) - g|^2 (Sussillo and Barak 2013), then count eigenvalues of the Jacobian
that are slow. The teacher has exactly one slow direction; how many does the method
report in a model fitted to its spikes?"""
import glob, json, re, sys
import numpy as np, torch
from teacher import load_teacher
from student_steer import load_student

def find_fp(W, b, starts, steps=1500, lr=0.02):
    z = torch.atanh(starts.clamp(-0.999, 0.999)).clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        g = torch.tanh(z); q = 0.5 * ((torch.tanh(g @ W.T + b) - g) ** 2).sum(1)
        opt.zero_grad(); q.sum().backward(); opt.step()
    g = torch.tanh(z).detach(); q = 0.5 * ((torch.tanh(g @ W.T + b) - g) ** 2).sum(1)
    return g, q.detach().numpy()

def counts(W, b, g, q, qmax=1e-5):
    rows = []
    Wn = W.numpy()
    for i in range(len(g)):
        if q[i] > qmax: continue
        f = np.tanh(Wn @ g[i].numpy() + b.numpy())
        ev = np.abs(np.linalg.eigvals((1 - f ** 2)[:, None] * Wn))
        rows.append([(ev > 0.8).sum(), (ev > 0.9).sum(), ((ev > 0.95) & (ev < 1.05)).sum(), (ev > 1.05).sum()])
    return rows

def summ(rows, N):
    if not rows: return dict(n_points=0)
    m = np.median(rows, 0)
    return dict(n_points=len(rows), N=N, slow08=float(m[0]), slow09=float(m[1]), marginal=float(m[2]), unstable=float(m[3]))

if __name__ == "__main__":
    torch.set_num_threads(1); out = {}
    for tag, dp, tp, pat in (("A", "runs/dataset.pt", "runs/teacher_g0.9_n0.05_s0.pt", r"runs/student_k4_lam[0-9.e-]+_s\d\.pt$"),
                             ("B", "runs/dataset_B.pt", "runs/teacherB_sustained_s0.pt", r"runs/student_k4_lam[0-9.e-]+_s\d_B\.pt$")):
        data = torch.load(dp, weights_only=False); tg = data["t_go"]; sel = list(range(0, 256, 8))
        net, _ = load_teacher(tp)
        H0 = torch.from_numpy(np.stack([data["H_teacher"][i, tg[i] - 15] for i in sel]))
        g, q = find_fp(net.W_rec.detach(), net.b.detach(), H0)
        out[tag] = dict(teacher=summ(counts(net.W_rec.detach(), net.b.detach(), g, q), 100), fits=[])
        print(tag, "TEACHER", out[tag]["teacher"], "median q %.1e" % np.median(q), flush=True)
        for p in sorted(glob.glob("runs/student_k4_lam*.pt")):
            if not re.match(pat, p): continue
            st, ck = load_student(p, data)
            if not np.isfinite(ck["eval"]["bps_heldin"]) or ck["eval"]["bps_heldin"] < 0: continue
            with torch.no_grad(): G, _ = st.generate(use_u=True)
            g, q = find_fp(st.W.detach(), st.b.detach(), torch.stack([G[i, tg[i] - 15] for i in sel]))
            r = summ(counts(st.W.detach(), st.b.detach(), g, q), st.N); r.update(lam=ck["lam_u"], seed=ck["seed"], q_median=float(np.median(q)))
            out[tag]["fits"].append(r); print(tag, r, flush=True)
    json.dump(out, open("runs/fp_count.json", "w"), indent=1)
