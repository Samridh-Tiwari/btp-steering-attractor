"""
Verification, not evaluation.

Low MSE on T=100 trials does NOT establish that the network built attractors.
A network with slowly leaky memory looks perfect over 100 steps and falls apart
over 1000. Since the whole project rests on the claim that these networks build
attractors, that claim gets tested directly.

Two tests:

  1. Qualitative: plot inputs, outputs and targets on a held-out trial.
     Does it look like a flip-flop?

  2. Memory persistence: drive the network with pulses for the first 50 steps,
     then feed exactly zero input for 950 more. A true attractor holds its state
     forever. A leaky solution decays. We measure the readout drift over that
     silent stretch. This is the test that separates the two, and it is the same
     logic as the 'run the generator autonomously' check from the SURF work.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import VanillaRNN
from tasks import flipflop

CKPT = "runs/flipflop_h100_g1.0_in1.0_reg0.0001_s0.pt"

ck = torch.load(CKPT, weights_only=False)
cfg = ck["cfg"]
net = VanillaRNN(3, cfg["n_hid"], 3, g=cfg["g"], input_scale=cfg["input_scale"])
net.load_state_dict(ck["state_dict"])
net.eval()

torch.manual_seed(7)

# ---------- test 1: does it look like a flip-flop ----------
u, z, m = flipflop(batch=1, T=100, p_pulse=0.05)
with torch.no_grad():
    y, H = net(u)

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
for b in range(3):
    ax = axes[b]
    ax.plot(z[0, :, b], color="0.6", lw=3, label="target" if b == 0 else None)
    ax.plot(y[0, :, b], color="C0", lw=1.5, label="RNN output" if b == 0 else None)
    pulses = u[0, :, b].nonzero().squeeze(-1)
    ax.vlines(pulses, -1.4, 1.4, color="C3", lw=1, alpha=.7,
              label="input pulse" if b == 0 else None)
    ax.set_ylim(-1.5, 1.5)
    ax.set_ylabel(f"bit {b}")
axes[0].legend(loc="upper right", fontsize=8, ncol=3)
axes[-1].set_xlabel("timestep")
fig.suptitle("3-bit flip-flop: held-out trial", fontsize=11)
fig.tight_layout()
fig.savefig("fig_flipflop_trial.png", dpi=140)
print("wrote fig_flipflop_trial.png")

# ---------- test 2: memory persistence over a long silence ----------
T_drive, T_silent = 50, 950
u_drive, _, _ = flipflop(batch=64, T=T_drive, p_pulse=0.10)
u_silent = torch.zeros(64, T_silent, 3)
u_long = torch.cat([u_drive, u_silent], dim=1)

with torch.no_grad():
    y_long, H_long = net(u_long)

y_at_silence_start = y_long[:, T_drive - 1]     # readout when input stops
y_at_end = y_long[:, -1]                        # readout 950 steps later
drift = (y_at_end - y_at_silence_start).abs()
state_drift = (H_long[:, -1] - H_long[:, T_drive - 1]).norm(dim=1)
state_norm = H_long[:, T_drive - 1].norm(dim=1)

print(f"\nmemory persistence over {T_silent} silent steps (64 trials):")
print(f"  mean |readout drift|        {drift.mean():.5f}")
print(f"  max  |readout drift|        {drift.max():.5f}")
print(f"  mean ||state drift|| / ||state||  {(state_drift / state_norm).mean():.5f}")
sign_kept = (torch.sign(y_at_end) == torch.sign(y_at_silence_start)).float().mean()
print(f"  fraction of bits keeping their sign  {sign_kept:.4f}")

fig, ax = plt.subplots(figsize=(10, 3.2))
for i in range(8):
    ax.plot(y_long[i, :, 0], lw=1, alpha=.8)
ax.axvline(T_drive, color="C3", ls="--", lw=1)
ax.text(T_drive + 12, 1.25, "input switched off", color="C3", fontsize=9)
ax.set_xlabel("timestep")
ax.set_ylabel("bit 0 output")
ax.set_ylim(-1.5, 1.5)
ax.set_title("memory persistence: 950 steps of zero input", fontsize=11)
fig.tight_layout()
fig.savefig("fig_flipflop_persistence.png", dpi=140)
print("wrote fig_flipflop_persistence.png")
