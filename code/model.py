"""
The network.

Deliberately a vanilla discrete-time tanh RNN, not a GRU/LSTM:

  h_{t+1} = tanh(W_rec h_t + W_in u_t + b)
  y_t     = W_out h_t + b_out

Why vanilla:
  1. The update map f(h, u) is one line, so the Jacobian df/dh is analytic and
     trivially correct. With a GRU you can still autodiff it, but debugging a
     wrong fixed point becomes guesswork.
  2. Sussillo & Barak 2013 characterised the flip-flop fixed-point structure in
     exactly this architecture, so their result is a valid ground truth to
     check against.

Why discrete time (as opposed to a continuous-time / Euler formulation):
  LFADS's generator is a discrete-time GRU, so in the SURF work fixed points
  were classified by |lambda| vs 1 on the discrete-time Jacobian. Keeping
  discrete time here means the stability criterion, the marginal-direction
  definition, and the controllability Gramian all carry over unchanged. Nothing
  has to be re-derived when comparing against that earlier result.

Stability criterion used throughout (discrete time):
  |lambda| < 1 - tol   -> stable    (perturbation decays)
  |lambda| > 1 + tol   -> unstable  (perturbation grows)
  otherwise            -> marginal  (perturbation neither decays nor grows)
"""

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, g=1.0, input_scale=1.0, seed=None):
        """
        g : spectral scale of the initial recurrent matrix. W_rec is initialised
            as g * N(0, 1/n_hid), so g < 1 starts in the decaying regime and
            g > 1 starts near the edge of chaos. This matters: initialisation
            regime biases which kind of solution training finds, and it is one
            of the knobs to sweep later.

        input_scale : scale of W_in at init. This is the lever that matters most
            for the BTP question -- how much authority the input channels have
            over the state. Kept explicit rather than buried in a default.
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.n_in, self.n_hid, self.n_out = n_in, n_hid, n_out

        self.W_rec = nn.Parameter(torch.randn(n_hid, n_hid) * (g / n_hid ** 0.5))
        self.W_in = nn.Parameter(torch.randn(n_hid, n_in) * (input_scale / n_in ** 0.5))
        self.b = nn.Parameter(torch.zeros(n_hid))
        self.W_out = nn.Parameter(torch.randn(n_out, n_hid) / n_hid ** 0.5)
        self.b_out = nn.Parameter(torch.zeros(n_out))

    def step(self, h, u):
        """One update. h: (batch, n_hid), u: (batch, n_in) -> (batch, n_hid)."""
        return torch.tanh(h @ self.W_rec.T + u @ self.W_in.T + self.b)

    def readout(self, h):
        return h @ self.W_out.T + self.b_out

    def forward(self, u, h0=None):
        """
        u : (batch, T, n_in)
        returns y (batch, T, n_out), H (batch, T, n_hid)
        H[:, t] is the state AFTER consuming u[:, t].
        """
        B, T, _ = u.shape
        h = torch.zeros(B, self.n_hid, device=u.device) if h0 is None else h0
        H = []
        for t in range(T):
            h = self.step(h, u[:, t])
            H.append(h)
        H = torch.stack(H, dim=1)
        return self.readout(H), H

    # ---- analysis helpers, used from week 4 on ----

    def f_autonomous(self, h):
        """The zero-input update map. Fixed points are solutions of f(h) = h."""
        return torch.tanh(h @ self.W_rec.T + self.b)

    def jacobian(self, h_star):
        """
        d f_autonomous / d h evaluated at h_star (n_hid,).

        For this architecture:
            f(h)_i = tanh(sum_j W_ij h_j + b_i)
            df_i/dh_j = (1 - f_i^2) * W_ij
        so the Jacobian is a row-rescaled copy of W_rec. Written out rather than
        autodiffed because it is exact, cheap, and easy to unit-test.
        """
        with torch.no_grad():
            pre = h_star @ self.W_rec.T + self.b
            d = 1.0 - torch.tanh(pre) ** 2          # (n_hid,)
            return d.unsqueeze(1) * self.W_rec       # (n_hid, n_hid)
