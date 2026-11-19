"""
Synthetic tasks for the BTP.

Two tasks, chosen for different jobs:

  FlipFlop  -- 3-bit memory. Discrete attractor task. Its answer is KNOWN
               (Sussillo & Barak 2013 found 8 stable fixed points, one per
               corner of the cube, plus saddles between them). We use it to
               validate the fixed-point finder. If our pipeline does not
               recover 8 stable fixed points here, the pipeline is broken and
               nothing downstream can be trusted.

  Integrator -- 1D velocity integration. Continuous attractor task. This is
               the scientific object of interest: to hold an arbitrary real
               number, the network must build a LINE of fixed points (a 1D
               continuous attractor), which gives marginal Jacobian
               directions. The controllability question lives here.

Conventions used everywhere in this project:
  u : inputs        (batch, T, n_in)
  z : targets       (batch, T, n_out)
  m : loss mask     (batch, T, n_out)   1 = count this timestep, 0 = ignore
Discrete time. One timestep = one network update. No physical units.
"""

import torch


def flipflop(batch=64, T=100, n_bits=3, p_pulse=0.05, device="cpu", generator=None):
    """
    3-bit flip-flop.

    Each of n_bits input channels independently delivers sparse +1/-1 pulses,
    one timestep long. Output channel i must hold the sign of the most recent
    pulse seen on input channel i, indefinitely, until the next pulse.

    Before a channel's first pulse the correct answer is undefined, so those
    timesteps are masked out of the loss. Leaving them unmasked would force
    the network to hold a 9th "output zero" state, which would show up as an
    extra fixed point and muddy the ground truth we are validating against.

    Returns u, z, m each of shape (batch, T, n_bits).
    """
    # pulse locations: 1 where a pulse happens
    hit = (torch.rand(batch, T, n_bits, device=device, generator=generator) < p_pulse)
    sign = torch.where(
        torch.rand(batch, T, n_bits, device=device, generator=generator) < 0.5, -1.0, 1.0
    )
    u = hit.float() * sign

    # target = sign of most recent nonzero input, held forward in time.
    # cummax over "index of last pulse" is the clean way to do this without a
    # python loop over T.
    idx = torch.arange(T, device=device).view(1, T, 1).expand(batch, T, n_bits)
    last_pulse_idx = torch.cummax(torch.where(hit, idx, torch.full_like(idx, -1)), dim=1).values
    seen = last_pulse_idx >= 0                      # has this channel pulsed yet?
    safe_idx = last_pulse_idx.clamp(min=0)
    z = torch.gather(u, 1, safe_idx) * seen.float()

    m = seen.float()
    return u, z, m


def integrator(batch=64, T=100, n_in=1, sigma=1.0, dt=0.05, clip=1.0,
               device="cpu", generator=None):
    """
    1D integrator / line attractor.

    Input is a continuous white-noise "velocity" signal. The output must be its
    running integral, i.e. z_t = clip(sum_{s<=t} dt * u_s). Holding an
    arbitrary real value with no input requires a continuum of stable states,
    so a trained network builds an approximate line attractor.

    clip bounds the target so the task stays inside the tanh dynamic range.
    Without it the target grows without bound and the network just learns a
    large gain instead of a genuine attractor.

    Every timestep is well defined here, so the mask is all ones (kept for a
    uniform interface with flipflop).

    Returns u, z, m each of shape (batch, T, n_in).
    """
    u = sigma * torch.randn(batch, T, n_in, device=device, generator=generator)
    z = torch.cumsum(dt * u, dim=1).clamp(-clip, clip)
    m = torch.ones_like(z)
    return u, z, m


TASKS = {"flipflop": flipflop, "integrator": integrator}
