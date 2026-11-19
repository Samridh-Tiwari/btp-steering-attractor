"""Two extra schematics for the talk: full LFADS as used here, and the GRU update gate."""
import diagrams as D
from diagrams import SVG, INK, MUTE, GRID, BLUE, ORANGE, GREEN, GREY, RED, LBLUE, LORANGE, LGREEN, LGREY, LRED

import os
D.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "diagrams")
os.makedirs(D.OUT, exist_ok=True)


def lfads():
    S = SVG(900, 400, "Full LFADS")
    S.text(450, 28, "Full LFADS (lfads-torch modules)", 17, weight="bold")
    S.box(20, 140, 120, 90, "Spikes", "one trial,\n50 neurons", fill=LGREY, stroke=MUTE, size=13)
    S.box(190, 60, 210, 80, "IC encoder", "reads the WHOLE trial\n(bidirectional GRU)", fill=LRED, stroke=RED, size=13, tcolor=RED)
    S.box(190, 230, 210, 80, "CI encoder", "bidirectional GRU,\nfeeds the controller", fill=LGREY, stroke=MUTE, size=13)
    S.arrow(140, 170, 187, 105, MUTE); S.arrow(140, 200, 187, 265, MUTE)
    S.box(450, 60, 170, 80, "Initial state g0", "sampled from\nits posterior", fill=LGREY, stroke=MUTE, size=13)
    S.box(450, 230, 170, 80, "Controller (GRU)", "sends u(t),\n4 channels", fill=LORANGE, stroke=ORANGE, size=13, tcolor=ORANGE)
    S.arrow(400, 100, 447, 100, MUTE); S.arrow(400, 270, 447, 270, MUTE)
    S.box(670, 140, 200, 90, "Generator (GRU)", "64 units, gated;\nholds the dynamics", fill=LBLUE, stroke=BLUE, size=13, tcolor=BLUE)
    S.arrow(620, 100, 690, 137, MUTE)
    S.arrow(620, 270, 690, 233, ORANGE, label="u(t)", lx=668, ly=268, lcolor=ORANGE)
    S.box(670, 300, 200, 70, "Factors to rates", "12 factors, Poisson\nreadout to spikes", fill=LGREY, stroke=MUTE, size=13)
    S.arrow(770, 230, 770, 297, MUTE)
    S.path("M 670 335 C 600 360, 560 345, 540 312", MUTE, 1.5, dashed=True, arrow=True)
    S.text(590, 372, "factors fed back", 11, color=MUTE)
    S.text(295, 165, "can write the plan into g0\nbefore the plan cue", 12, color=RED)
    S.text(535, 215, "KL penalty (kl_co_scale)", 11, color=ORANGE)
    S.save("d16_lfads")


def gru_gate():
    S = SVG(900, 380, "The GRU update gate")
    S.text(450, 28, "One generator unit: the gate that keeps a value also blocks the input", 17, weight="bold")
    S.text(450, 62, "new value  =  z × old value  +  (1 − z) × candidate(input)", 16, color=INK)
    # left: gate open
    S.box(40, 95, 380, 255, None, None, fill="#ffffff", stroke=GRID)
    S.text(230, 125, "Gate open (z ≈ 0)", 15, weight="bold", color=ORANGE)
    S.box(70, 160, 120, 60, "input", "u(t)", fill=LORANGE, stroke=ORANGE, size=13, tcolor=ORANGE)
    S.box(270, 160, 120, 60, "unit", "value changes", fill=LGREY, stroke=MUTE, size=13)
    S.arrow(190, 190, 267, 190, ORANGE, sw=4)
    S.text(230, 260, "the input gets in,\nbut the unit forgets its old value", 13, color=INK)
    S.text(230, 320, "fast units: the controller's push lands here", 12, color=ORANGE)
    # right: gate closed
    S.box(480, 95, 380, 255, None, None, fill="#ffffff", stroke=GRID)
    S.text(670, 125, "Gate closed (z ≈ 1)", 15, weight="bold", color=BLUE)
    S.box(510, 160, 120, 60, "input", "u(t)", fill=LORANGE, stroke=ORANGE, size=13, tcolor=ORANGE)
    S.box(710, 160, 120, 60, "unit", "holds the plan", fill=LBLUE, stroke=BLUE, size=13, tcolor=BLUE)
    S.arrow(630, 190, 668, 190, ORANGE, sw=4)
    S.line(675, 172, 675, 208, RED, 4)
    S.text(670, 260, "the unit keeps its value (memory),\nand the input is shut out", 13, color=INK)
    S.text(670, 320, "memory units: 74–98% of the sensitivity, ≤3% of the push", 12, color=BLUE)
    S.save("d17_gru_gate")


lfads(); gru_gate()
