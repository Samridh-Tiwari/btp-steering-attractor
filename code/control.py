"""
Controllability of a behavioural readout through a given set of input channels.

THE SETUP
---------
Linearise the network at an operating point h* (a delay-period state, where
the plan is being held):

    δh(t+1) = A δh(t) + D B u(t)          A = D W,  D = diag(1 - f²)
    δy(t)   = cᵀ δh(t)                     c = the behavioural readout

B is the set of input channels (N x k). D appears because input enters before
the tanh, so its effect is scaled by the local gain.

THE ONE NUMBER THAT MATTERS
---------------------------
Drive for T steps, starting from δh = 0. The readout at the end is

    δy(T) = Σ_s cᵀ A^(T-1-s) D B u(s)

The OUTPUT CONTROLLABILITY GRAMIAN is

    G(B) = Σ_t ‖Bᵀ D (Aᵀ)^t c‖²  =  tr(Bᵀ Q̃ B),     Q̃ = D Q D,
    Q    = Σ_{t=0}^{T-1} (Aᵀ)^t c cᵀ A^t

G is how much the readout moves per unit of input energy.

WHY IT IS EXACTLY THE LQR QUESTION
----------------------------------
Minimise (δy(T) - e)² + ρ Σ‖u(s)‖² — miss the target by as little as possible
while paying ρ per unit of input energy. The optimum is closed form and the
fraction of the error it removes is exactly

    error reduction = G / (G + ρ)

So the SURF "0.1% error reduction" means G was about 0.001·ρ. Everything
about steering a scalar behaviour through B is contained in G.

COMPARING CHANNEL SETS FAIRLY
-----------------------------
G depends on two things: where the channels POINT, and how STRONG they are.
To isolate aim, every channel set is reduced to an orthonormal basis P of its
column span (k unit-norm, mutually orthogonal channels), so

    G(P) = tr(Pᵀ Q̃ P)

and the comparisons are:
    fit      — span of the model's own inferred-input matrix
    random   — random k-dimensional subspaces (many draws)
    optimal  — the best possible k-dimensional subspace: top-k eigenvectors
               of Q̃, giving G = sum of the top-k eigenvalues
    full     — all N directions actuated: G = tr(Q̃)

    aim score = (G_fit - mean G_random) / (G_opt - mean G_random)
                0 = no better than chance, 1 = optimal, < 0 = worse than chance

    fraction of full = G_opt(k) / G_full
                how much of complete actuation k well-aimed channels recover.
                This is the direct test of "k channels are too few".
"""

import numpy as np


def output_obs_gramian(A, c, T):
    """Q = Σ_{t<T} (Aᵀ)^t c cᵀ A^t, built by iterating v ← Aᵀ v."""
    v = np.asarray(c, float).copy()
    Q = np.zeros((len(v), len(v)))
    At = A.T
    for _ in range(T):
        Q += np.outer(v, v)
        v = At @ v
    return Q


def orth(B):
    """Orthonormal basis for the column span of B."""
    q, r = np.linalg.qr(np.asarray(B, float))
    keep = np.abs(np.diag(r)) > 1e-10 * np.abs(np.diag(r)).max()
    return q[:, keep]


def G_of(Qt, P):
    return float(np.trace(P.T @ Qt @ P))


def G_opt(Qt, k):
    ev = np.linalg.eigvalsh((Qt + Qt.T) / 2)[::-1]
    return float(ev[:k].sum())


def G_random(Qt, k, n_draws=200, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    N = Qt.shape[0]
    out = np.empty(n_draws)
    for i in range(n_draws):
        P = np.linalg.qr(rng.standard_normal((N, k)))[0]
        out[i] = G_of(Qt, P)
    return out


def slow_subspace(A, m):
    """Orthonormal basis for the span of the m slowest eigen-directions of A
    (complex pairs contribute their real and imaginary parts)."""
    ev, V = np.linalg.eig(A)
    o = np.argsort(-np.abs(ev))
    cols, used = [], 0
    for i in o:
        if used >= m:
            break
        if abs(ev[i].imag) < 1e-9:
            cols.append(V[:, i].real); used += 1
        elif ev[i].imag > 0:
            cols += [V[:, i].real, V[:, i].imag]; used += 2
    return orth(np.array(cols).T), np.abs(ev[o])


def analyse(A, D, c, channel_sets, T=25, n_random=200, rng=None, k_list=None):
    """
    A: Jacobian (N x N) at the operating point.  D: diag gain vector (N,).
    c: readout (N,).  channel_sets: dict name -> N x k matrix.
    Returns a dict of results per channel set plus random/optimal references.
    """
    Q = output_obs_gramian(A, c, T)
    Qt = (D[:, None] * Q) * D[None, :]
    G_full = float(np.trace(Qt))
    res = {"G_full": G_full, "sets": {}}
    for name, B in channel_sets.items():
        P = orth(B)
        k = P.shape[1]
        g_fit = G_of(Qt, P)
        g_rand = G_random(Qt, k, n_random, rng)
        g_opt = G_opt(Qt, k)
        denom = g_opt - g_rand.mean()
        res["sets"][name] = dict(
            k=k, G_fit=g_fit, G_rand_mean=float(g_rand.mean()),
            G_rand_p95=float(np.percentile(g_rand, 95)),
            pct_of_random=float((g_rand < g_fit).mean() * 100),
            G_opt=g_opt, aim=float((g_fit - g_rand.mean()) / denom) if denom > 0 else np.nan,
            frac_full_fit=g_fit / G_full, frac_full_opt=g_opt / G_full)
    if k_list is not None:
        ev = np.linalg.eigvalsh((Qt + Qt.T) / 2)[::-1]
        res["opt_curve"] = {int(k): float(ev[:k].sum() / G_full) for k in k_list}
        res["rand_curve"] = {int(k): float(k / len(c)) for k in k_list}
    return res


def error_reduction(G, rho):
    return G / (G + rho)


# ---------------------------------------------------------------------------
# self-test: the closed form must match a brute-force least-squares controller
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N, k, T, rho = 12, 3, 8, 0.7
    A = rng.standard_normal((N, N)) * 0.3
    D = rng.uniform(0.3, 1.0, N)
    B = rng.standard_normal((N, k))
    c = rng.standard_normal(N)
    e = 1.3                                    # error to remove
    # brute force: stack the map u -> δy(T), solve the regularised problem
    Mcols = []
    for s in range(T):
        Ap = np.linalg.matrix_power(A, T - 1 - s)
        Mcols.append(c @ Ap @ (D[:, None] * B))
    g = np.concatenate(Mcols)                  # δy = g · u
    u = -g * e / (g @ g + rho)
    resid = e + g @ u
    bf_reduction = 1 - resid / e
    Q = output_obs_gramian(A, c, T)
    Qt = (D[:, None] * Q) * D[None, :]
    G = np.trace(B.T @ Qt @ B)
    print(f"brute-force error reduction {bf_reduction:.6f}")
    print(f"closed form G/(G+rho)       {error_reduction(G, rho):.6f}")
    print(f"G from Gramian {G:.6f}   |g|^2 {g @ g:.6f}")
    # optimal subspace really is optimal: beats 2000 random orthonormal sets
    Gr = G_random(Qt, k, 2000, rng)
    print(f"G_opt {G_opt(Qt, k):.4f}   max random {Gr.max():.4f}   mean random {Gr.mean():.4f}"
          f"   expected k/N*tr {k / N * np.trace(Qt):.4f}")
