"""
Characterising a continuous attractor WITHOUT running an optimiser.

THE PROBLEM THIS SOLVES
-----------------------
fixedpoints.py finds fixed points by descending q(h) = 0.5*||f(h) - h||^2. On a
continuous attractor that fails, because a trained manifold is never exactly
flat: q has a small gradient ALONG it, so the optimiser slides every candidate
down the manifold until it falls off the ends. It then reports the handful of
genuine fixed points at those ends and claims there is no attractor. The
flip-flop validation cannot catch this, since discrete attractors have no
manifold to slide along.

THE APPROACH HERE
-----------------
Skip the optimiser. Let the network put itself on the manifold:

    drive it on the task -> cut the input -> let it relax autonomously

After relaxation the transients (the strongly contracting directions) have
died, so whatever spread remains lies on the attractor. Those states are where
the network genuinely operates, which is also the honest place to ask about
controllability later. Take the Jacobian there.

THE TEST THAT DECIDES WHETHER IT WORKED
---------------------------------------
Two independent estimates of the same object must agree:

  (a) LOCAL, from dynamics: at each state, eigenvalues of J with |lambda| ~ 1.
      Their eigenvectors span the directions in which the dynamics neither
      contract nor expand.

  (b) GLOBAL, from geometry: PCA of the whole settled cloud. Its top PCs span
      the directions the manifold actually extends in.

If the network has a d-dimensional attractor manifold, these must be the same
d-dimensional subspace. Measuring their agreement (principal angles) is the
check. Two estimates from unrelated computations agreeing is evidence; either
one alone is not.

CONTROL
-------
Run the same analysis on the flip-flop network. Its memory states are isolated
points, so it should show ~0 marginal directions. If this code reports marginal
directions there too, the marginal criterion is just mislabelling slow
contraction and means nothing.

TOLERANCE
---------
"Marginal" means |lambda| within tol of 1, and tol is a judgement call, not a
fact about the network. Every marginal count here is therefore reported across
a sweep of tol. A result that only exists at one tol is not a result.
"""

import argparse
import numpy as np
import torch
from scipy.linalg import subspace_angles

from model import VanillaRNN
from tasks import TASKS


# ----------------------------------------------------------------------------

def load(ckpt):
    ck = torch.load(ckpt, weights_only=False)
    c = ck["cfg"]
    probe_u, probe_z, _ = TASKS[c["task"]](batch=2, T=8)
    net = VanillaRNN(probe_u.shape[-1], c["n_hid"], probe_z.shape[-1],
                     g=c["g"], input_scale=c["input_scale"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, c


def settled_states(net, task, batch=256, T=200, settle=2000, seed=3):
    """
    Put the network on its attractor and return the states it lands on.

    Drive it on the task, then evolve autonomously for `settle` steps. The
    contracting directions collapse during that stretch; what is left is spread
    along the attractor itself.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        u, _, _ = TASKS[task](batch=batch, T=T)
        _, H = net(u)
        h = H[:, -1].clone()
        for _ in range(settle):
            h = net.f_autonomous(h)
        speed = (net.f_autonomous(h) - h).norm(dim=1)
    return h, speed


# ----------------------------------------------------------------------------

def marginal_basis(net, h_star, tol):
    """
    Real orthonormal basis for the marginal subspace at one state.

    J is not symmetric, so eigenvalues come in complex conjugate pairs. A pair
    does not correspond to two independent complex directions but to one real
    2D invariant plane, spanned by Re(v) and Im(v). Building the real basis this
    way (rather than naively stacking complex eigenvectors) is what makes the
    subspace comparison against PCA meaningful.
    """
    J = net.jacobian(h_star)
    vals, vecs = torch.linalg.eig(J)
    mag = vals.abs()
    sel = (mag >= 1 - tol) & (mag <= 1 + tol)
    if sel.sum() == 0:
        return None, vals

    cols = []
    for i in torch.nonzero(sel).squeeze(-1).tolist():
        v = vecs[:, i]
        if vals[i].imag.abs() < 1e-12:
            cols.append(v.real)
        else:
            cols.append(v.real)
            cols.append(v.imag)
    B = torch.stack(cols, dim=1).double()
    # orthonormalise; conjugate pairs contribute duplicate planes, so drop
    # numerically dependent columns via the rank-revealing QR-equivalent
    U, S, _ = torch.linalg.svd(B, full_matrices=False)
    keep = S > 1e-8 * S[0]
    return U[:, keep].numpy(), vals


def tangent_basis(H, d):
    """Top-d PCA directions of the settled cloud: the manifold's tangent space."""
    Hc = (H - H.mean(0)).double()
    _, _, V = torch.linalg.svd(Hc, full_matrices=False)
    return V[:d].T.numpy()


# ----------------------------------------------------------------------------

def analyse(ckpt, settle=2000, n_probe=40, tols=(3e-4, 1e-3, 3e-3, 1e-2),
            batch=256, T=200, seed=3):

    net, cfg = load(ckpt)
    task = cfg["task"]
    print(f"network: {task}  N={cfg['n_hid']}  g={cfg['g']}  "
          f"input_scale={cfg['input_scale']}  reg_h={cfg['reg_h']}")

    H, speed = settled_states(net, task, batch=batch, T=T, settle=settle, seed=seed)
    print(f"\nsettled {batch} states after {settle} autonomous steps")
    print(f"  speed ||f(h)-h||:  median {speed.median():.3e}   max {speed.max():.3e}")

    # --- geometry: how many dimensions does the cloud occupy ---
    Hc = H - H.mean(0)
    S = torch.linalg.svdvals(Hc)
    var = ((S ** 2) / (S ** 2).sum()).numpy()
    cum = np.cumsum(var)
    d_geom = int(np.searchsorted(cum, 0.90) + 1)
    print(f"\nGEOMETRY (PCA of the settled cloud)")
    print(f"  PC variance: {np.round(var[:6], 4)}")
    print(f"  dimensions for 90% variance: {d_geom}")

    # --- dynamics: marginal direction counts, swept over tolerance ---
    idx = torch.randperm(H.shape[0])[:n_probe]
    print(f"\nDYNAMICS (Jacobian at {n_probe} of those states)")
    print(f"  {'tol':>8}   {'marginal dirs (mean+-sd)':>26}   {'unstable':>9}   {'|l|max':>8}")
    counts_by_tol = {}
    for tol in tols:
        ms, us, mx = [], [], []
        for i in idx:
            J = net.jacobian(H[i])
            mag = torch.linalg.eigvals(J).abs()
            ms.append(int(((mag >= 1 - tol) & (mag <= 1 + tol)).sum()))
            us.append(int((mag > 1 + tol).sum()))
            mx.append(float(mag.max()))
        counts_by_tol[tol] = float(np.mean(ms))
        print(f"  {tol:8.1e}   {np.mean(ms):12.2f} +- {np.std(ms):<10.2f}   "
              f"{np.mean(us):9.2f}   {np.mean(mx):8.4f}")

    # --- the agreement test ---
    print(f"\nAGREEMENT (marginal eigenvector subspace  vs  PCA tangent space)")
    Tb = tangent_basis(H, d_geom)
    print(f"  tangent space = top {d_geom} PCs")
    print(f"  {'tol':>8}   {'dim(marg)':>9}   {'principal angles (deg)':>34}   {'mean':>6}")
    for tol in tols:
        Bs, dims = [], []
        for i in idx[:12]:
            B, _ = marginal_basis(net, H[i], tol)
            if B is None:
                continue
            k = min(B.shape[1], Tb.shape[1])
            ang = np.degrees(subspace_angles(B[:, :k], Tb[:, :k]))
            Bs.append(ang)
            dims.append(B.shape[1])
        if not Bs:
            print(f"  {tol:8.1e}   {'0':>9}   (no marginal directions found)")
            continue
        A = np.mean(np.array(Bs), axis=0)
        print(f"  {tol:8.1e}   {np.mean(dims):9.1f}   {np.array2string(np.round(A,1), max_line_width=200):>34}   "
              f"{A.mean():6.1f}")
    print("\n  small angles (<~20 deg) => the two estimates describe the same subspace")
    print("  angles near 90 deg      => they are unrelated, and the marginal count is not the manifold")

    return dict(net=net, H=H, d_geom=d_geom, counts=counts_by_tol)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--settle", type=int, default=2000)
    p.add_argument("--n_probe", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--seed", type=int, default=3)
    analyse(**vars(p.parse_args()))
