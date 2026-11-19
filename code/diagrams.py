"""Explanatory schematics for the BTP report (SVG). Written for a reader with little background."""
import math, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "diagrams")
os.makedirs(OUT, exist_ok=True)

INK, MUTE, GRID = "#1f1f1f", "#6b6b66", "#d9d8d3"
BLUE, ORANGE, GREEN, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a95", "#c8433a"
LBLUE, LORANGE, LGREEN, LGREY, LRED = "#e7f1fb", "#fbeee7", "#e5f4ec", "#f0efec", "#fbebeb"
FONT = "DejaVu Sans, Arial, sans-serif"


class SVG:
    def __init__(self, w, h, label):
        self.w, self.h, self.label, self.parts = w, h, label, []
        self.markers = set()

    def add(self, s):
        self.parts.append(s)

    def marker(self, color):
        mid = "a" + color.strip("#")
        self.markers.add((mid, color))
        return mid

    def text(self, x, y, s, size=15, anchor="middle", weight="normal", color=INK, italic=False):
        lines = s.split("\n") if isinstance(s, str) else s
        st = ' font-style="italic"' if italic else ""
        for i, ln in enumerate(lines):
            self.add(f'<text x="{x}" y="{y + i * size * 1.25:.1f}" font-family="{FONT}" font-size="{size}" '
                     f'text-anchor="{anchor}" font-weight="{weight}" fill="{color}"{st}>{esc(ln)}</text>')

    def box(self, x, y, w, h, title=None, body=None, fill=LGREY, stroke=MUTE, size=14, r=8, tcolor=INK, sw=1.5):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        lines_t = title.split("\n") if title else []
        lines_b = body.split("\n") if body else []
        n = len(lines_t) * (size + 1) * 1.25 + len(lines_b) * size * 1.25
        cy = y + h / 2 - n / 2 + size
        for ln in lines_t:
            self.text(x + w / 2, cy, ln, size + 1, weight="bold", color=tcolor); cy += (size + 1) * 1.25
        for ln in lines_b:
            self.text(x + w / 2, cy, ln, size, color=INK); cy += size * 1.25

    def arrow(self, x1, y1, x2, y2, color=INK, label=None, dashed=False, sw=2, lx=None, ly=None, lsize=13,
              lanchor="middle", lcolor=None):
        mid = self.marker(color)
        d = ' stroke-dasharray="6 5"' if dashed else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{mid})"/>')
        if label:
            self.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 7,
                      label, lsize, anchor=lanchor, color=lcolor or MUTE)

    def path(self, d, color=INK, sw=2, fill="none", dashed=False, arrow=False):
        a = f' marker-end="url(#{self.marker(color)})"' if arrow else ""
        ds = ' stroke-dasharray="6 5"' if dashed else ""
        self.add(f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="{fill}"{ds}{a}/>')

    def line(self, x1, y1, x2, y2, color=INK, sw=1.5, dashed=False):
        ds = ' stroke-dasharray="5 4"' if dashed else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{ds}/>')

    def circle(self, x, y, r, fill=INK, stroke="none", sw=1.5):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def save(self, name):
        defs = "".join(
            f'<marker id="{m}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for m, c in sorted(self.markers))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{esc(self.label)}">'
               f'<defs>{defs}</defs><rect width="{self.w}" height="{self.h}" fill="white"/>' + "".join(self.parts) + "</svg>")
        open(f"{OUT}/{name}.svg", "w").write(svg)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def chip(S, x, y, w, h, s, fill, color):
    S.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" fill="{fill}" stroke="{color}" stroke-width="1.5"/>')
    S.text(x + w / 2, y + h / 2 + 5, s, 14, weight="bold", color=color)


# ---------------------------------------------------------------- D1: the story
def d1():
    S = SVG(900, 250, "The SURF result and the question this project asks")
    xs = [15, 192, 369, 546, 723]
    items = [("Mouse task", "lever pull after\na cue (Purdue)"), ("Recording", "neurons in\nforelimb M1"),
             ("LFADS model", "fit to the\nspikes"), ("Attractor found", "holds the plan\nfor the pull"),
             ("Steering test", "LQR through 4\ninferred inputs")]
    for x, (t, b) in zip(xs, items):
        S.box(x, 25, 162, 82, t, b, fill=LBLUE, stroke=BLUE)
    for a, b in zip(xs[:-1], xs[1:]):
        S.arrow(a + 162, 66, b - 3, 66)
    S.box(723, 150, 162, 70, "Result", "error reduced\nby only 0.1%", fill=LRED, stroke=RED, tcolor=RED)
    S.arrow(804, 107, 804, 147)
    S.box(369, 150, 316, 70, "SURF explanation", "4 channels are too few for a\n128-D state (never tested)", fill=LGREY)
    S.arrow(723, 185, 688, 185)
    S.box(15, 150, 316, 70, "This BTP", "test that explanation and four\nothers, with no lab data", fill=LGREEN, stroke=GREEN, tcolor="#11795a")
    S.arrow(369, 185, 334, 185)
    S.save("d1_story")


# ---------------------------------------------------------------- D2: boat hypotheses
def boat(S, x, y, scale=1.0):
    w, h = 70 * scale, 30 * scale
    S.add(f'<path d="M{x-w/2},{y-h/2} L{x+w/4},{y-h/2} L{x+w/2},{y} L{x+w/4},{y+h/2} L{x-w/2},{y+h/2} Z" '
          f'fill="#ffffff" stroke="{INK}" stroke-width="2"/>')


def flag(S, x, y):
    S.line(x, y, x, y - 34, INK, 2)
    S.add(f'<path d="M{x},{y-34} L{x+22},{y-27} L{x},{y-20} Z" fill="{RED}"/>')
    S.text(x, y + 18, "target plan", 13, color=MUTE)


def d2():
    S = SVG(900, 400, "Three explanations for the failed steering, drawn as a boat with thrusters")
    titles = ["1. Too few thrusters", "2. Thrusters aimed wrong", "3. Built-in engines push\nthe wrong way"]
    hyp = ["Dimension", "Parameterisation", "Geometry"]
    preds = ["more channels fix it", "same number, aimed\nproperly, fixes it", "even the real inputs\nsteer the plan poorly"]
    for i in range(3):
        x0 = 15 + i * 295
        S.add(f'<rect x="{x0}" y="10" width="280" height="325" rx="10" fill="#fbfbfa" stroke="{GRID}"/>')
        S.text(x0 + 140, 34, hyp[i], 13, color=MUTE)
        S.text(x0 + 140, 54, titles[i], 16, weight="bold")
    # panel 1: one tiny thruster, falls short
    x0 = 15
    boat(S, x0 + 70, 160); flag(S, x0 + 235, 175)
    S.arrow(x0 + 108, 160, x0 + 140, 160, ORANGE, sw=3)
    S.path(f"M{x0+145},160 L{x0+190},160", MUTE, 1.5, dashed=True)
    S.text(x0 + 140, 215, "1 weak thruster: boat moves", 13, color=INK)
    S.text(x0 + 140, 232, "a little, stops short", 13, color=INK)
    # panel 2: four thrusters pointing sideways/back
    x0 = 310
    boat(S, x0 + 70, 160); flag(S, x0 + 235, 175)
    for x1, y1, x2, y2 in ((70, 145, 70, 112), (70, 175, 70, 208), (36, 150, 12, 124), (36, 170, 12, 196)):
        S.arrow(x0 + x1, y1, x0 + x2, y2, ORANGE, sw=3)
    S.text(x0 + 140, 232, "4 thrusters, none pointing", 13)
    S.text(x0 + 140, 249, "at the target", 13)
    # panel 3: canal with current pushing back across
    x0 = 605
    S.line(x0 + 20, 125, x0 + 260, 125, MUTE, 3); S.line(x0 + 20, 195, x0 + 260, 195, MUTE, 3)
    S.text(x0 + 140, 117, "sloping canal banks", 12, color=MUTE)
    boat(S, x0 + 75, 160, 0.85)
    S.line(x0 + 235, 182, x0 + 235, 148, INK, 2)
    S.add(f'<path d="M{x0+235},148 L{x0+257},155 L{x0+235},162 Z" fill="{RED}"/>')
    S.text(x0 + 235, 213, "target plan", 13, color=MUTE)
    S.arrow(x0 + 70, 174, x0 + 70, 192, ORANGE, sw=3); S.arrow(x0 + 70, 146, x0 + 70, 128, ORANGE, sw=3)
    S.path(f"M{x0+92},131 q16,14 4,26", BLUE, 1.6, arrow=True)
    S.path(f"M{x0+92},189 q16,-14 4,-26", BLUE, 1.6, arrow=True)
    S.text(x0 + 150, 232, "pushes across the canal", 13)
    S.text(x0 + 150, 249, "fade, only pushes along it last", 13)
    for i in range(3):
        x0 = 15 + i * 295
        S.line(x0 + 20, 272, x0 + 260, 272, GRID, 1)
        S.text(x0 + 140, 293, "Prediction:", 13, weight="bold", color=MUTE)
        S.text(x0 + 140, 311, preds[i], 13)
    S.add(f'<rect x="15" y="348" width="870" height="44" rx="8" fill="{LBLUE}" stroke="{BLUE}"/>')
    S.text(450, 366, "Read the boat as moving through thick honey: it has no inertia, so it moves only while a thruster", 13)
    S.text(450, 383, "pushes and stays wherever it stops (along the canal). Pushed towards a bank, it slides back to the middle.", 13)
    S.save("d2_boats")


# ---------------------------------------------------------------- D3: attractor intuition
def d3():
    S = SVG(900, 300, "A line attractor: the plan is a position along a flat valley")
    # left: top view, curved line coloured by plan
    S.text(150, 28, "Top view: one point per plan", 15, weight="bold")
    cols = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    pts = [(40 + i * 37, 200 - 90 * math.sin(math.pi * i / 6) ** 1.0 * 0.0 - 0.55 * (i - 3) ** 2 * 10) for i in range(7)]
    S.path("M" + " L".join(f"{x},{y}" for x, y in pts), GRID, 10)
    for (x, y), c in zip(pts, cols):
        S.circle(x, y, 9, c, INK, 1)
    S.text(pts[0][0] + 10, pts[0][1] - 20, "weak pull", 13, color=MUTE)
    S.text(pts[-1][0] - 10, pts[-1][1] - 20, "strong pull", 13, color=MUTE)
    S.text(150, 270, "each trial parks the state somewhere", 13)
    S.text(150, 287, "on this line and it stays there", 13)
    # middle: along the valley (flat)
    S.text(450, 28, "Along the line: flat", 15, weight="bold")
    S.line(330, 180, 570, 180, INK, 3)
    S.circle(390, 166, 13, BLUE)
    S.add(f'<circle cx="490" cy="166" r="13" fill="none" stroke="{BLUE}" stroke-width="2" stroke-dasharray="4 3"/>')
    S.arrow(408, 140, 475, 140, ORANGE, "push", sw=3, lcolor=ORANGE)
    S.text(450, 225, "a push moves the plan", 13)
    S.text(450, 242, "and it stays moved (memory)", 13)
    # right: across the valley (steep)
    S.text(750, 28, "Across the line: steep", 15, weight="bold")
    S.path("M640,80 Q750,330 860,80", INK, 3)
    S.circle(750, 191, 13, BLUE)
    S.add(f'<circle cx="700" cy="150" r="13" fill="none" stroke="{BLUE}" stroke-width="2" stroke-dasharray="4 3"/>')
    S.path("M706,132 Q740,135 748,170", RED, 2, arrow=True)
    S.text(750, 245, "a push across rolls back", 13)
    S.text(750, 262, "within a few steps", 13)
    S.save("d3_attractor")


# ---------------------------------------------------------------- D4: task timeline
def d4():
    S = SVG(900, 390, "Task timelines: teacher, monkey and the SURF mouse task")
    def row(y, name, a, b, c, d, plan=True, same=False, fill=LBLUE):
        S.text(15, y - 38, name, 15, anchor="start", weight="bold")
        S.add(f'<rect x="250" y="{y-28}" width="330" height="40" fill="{fill}" stroke="none"/>')
        S.line(110, y + 12, 880, y + 12, INK, 1.5)
        if plan:
            S.add(f'<rect x="200" y="{y-28}" width="16" height="40" fill="{ORANGE}"/>')
            S.text(208, y + 32, a, 13, color=ORANGE)
        else:
            S.text(208, y + 32, a, 13, color=MUTE)
        S.add(f'<rect x="590" y="{y-28}" width="10" height="40" fill="{GREEN}"/>')
        S.text(415, y - 3, b, 13, color=BLUE if plan else MUTE)
        S.text(595, y + 32, c, 13, color="#11795a")
        amps = (20, 20, 20) if same else (10, 20, 30)
        for k, amp in enumerate(amps):
            S.path(f"M620,{y+12} L650,{y+12} L660,{y+12-amp} L720,{y+12-amp} L730,{y+12} L760,{y+12}",
                   ["#9ec5f4", "#3987e5", "#0d366b"][k], 2.2)
        S.text(820, y - 22, d, 13)
    row(95, "Teacher network (synthetic, idealised)", "value pulse\n(sets plan v)", "delay: hold the plan, no input", "go cue", "pull of\nsize v")
    row(215, "Monkey (MC_Maze, public data)", "target\nappears", "delay: hold the plan, no input", "go cue", "reach to\nthe target")
    row(335, "Mouse (SURF task)", "no plan cue", "waiting since the last pull (~3.7 s apart)", "tone = go", "the same\npull each time",
        plan=False, same=True, fill=LGREY)
    S.save("d4_task")


# ---------------------------------------------------------------- D5: pipeline
def d5():
    S = SVG(900, 345, "The pipeline: the same analysis is run where the answer is known and on public data")
    S.text(15, 22, "Synthetic: answer known", 14, anchor="start", weight="bold", color=BLUE)
    S.box(15, 35, 130, 78, "Task", "load, hold,\npull", fill=LBLUE, stroke=BLUE)
    S.box(175, 35, 150, 78, "Teacher", "network (100\nunits), wiring\nknown", fill=LBLUE, stroke=BLUE)
    S.box(355, 35, 135, 78, "Spikes", "60 neurons\n50 fit\n+ 10 held out", fill=LBLUE, stroke=BLUE)
    S.box(520, 35, 150, 78, "Student", "LFADS-style\nmodel, sees\nonly spikes", fill=LORANGE, stroke=ORANGE)
    S.arrow(145, 74, 172, 74); S.arrow(325, 74, 352, 74)
    S.arrow(490, 74, 517, 74, label="fit")
    S.text(15, 222, "Public data: answer unknown", 14, anchor="start", weight="bold", color="#11795a")
    S.box(15, 235, 310, 78, "Monkey motor cortex", "MC_Maze, 366 trials,\n137 neurons", fill=LGREEN, stroke=GREEN)
    S.box(355, 235, 135, 78, "Spikes", "137 neurons\n103 fit\n+ 34 held out", fill=LGREEN, stroke=GREEN)
    S.box(520, 235, 150, 78, "Student", "same model,\nsame settings", fill=LORANGE, stroke=ORANGE)
    S.arrow(325, 274, 352, 274); S.arrow(490, 274, 517, 274, label="fit")
    S.box(705, 35, 180, 278, "Steering analysis", "authority of:\n\nits own inferred\nchannels\n\nvs random\nchannels\n\nvs the best possible\nchannels",
          fill="#ffffff", stroke=INK)
    S.arrow(670, 74, 702, 74); S.arrow(670, 274, 702, 274)
    S.path("M250,113 L250,165 L690,165", BLUE, 2, dashed=True, arrow=True)
    S.text(470, 157, "true answer, used to check every step", 13, color=BLUE)
    S.save("d5_pipeline")


# ---------------------------------------------------------------- D6: student anatomy
def d6():
    S = SVG(900, 290, "Inside the student model")
    S.box(10, 95, 150, 95, "Controller", "inferred inputs u(t)\none number per\nchannel per step", fill=LORANGE, stroke=ORANGE)
    S.box(250, 95, 150, 95, "Generator", "recurrent network\n(64 units), its own\ndynamics W", fill=LBLUE, stroke=BLUE)
    S.box(470, 95, 120, 95, "Factors", "12 slow\nsignals", fill=LGREY)
    S.box(660, 95, 225, 95, "Firing rates", "one per neuron,\ncompared with the\nrecorded spikes", fill=LGREY)
    S.arrow(160, 142, 247, 142, ORANGE, "B: k channels", lcolor=ORANGE, ly=132, lsize=12)
    S.arrow(400, 142, 467, 142); S.arrow(590, 142, 657, 142)
    S.path("M300,95 C300,50 350,50 350,95", BLUE, 2, arrow=True)
    S.text(325, 45, "W: next state", 13, color=BLUE)
    S.box(250, 220, 150, 50, None, "initial state g(0)\nper trial", fill="#ffffff", stroke=MUTE, size=13)
    S.arrow(325, 220, 325, 193)
    S.text(120, 230, "penalty on using u", 13, color=ORANGE, weight="bold")
    S.text(120, 248, "small: controller explains a lot", 13, color=MUTE)
    S.text(120, 266, "large: dynamics must do the work", 13, color=MUTE)
    S.text(772, 230, "training adjusts everything", 13, color=MUTE)
    S.text(772, 248, "so the rates match the spikes", 13, color=MUTE)
    S.save("d6_student")


# ---------------------------------------------------------------- D7: authority
def d7():
    S = SVG(900, 330, "Authority: how much of a channel's push lands on the direction that changes behaviour")
    ox, oy, L = 90, 250, 190
    S.text(210, 26, "One channel = one push direction", 15, weight="bold")
    S.line(ox - 20, oy, ox + 260, oy, GRID, 1); S.line(ox, oy + 20, ox, oy - 220, GRID, 1)
    # sensitive direction s along x
    S.arrow(ox, oy, ox + 250, oy, BLUE, sw=5)
    S.text(ox + 250, oy + 26, "direction that changes the pull", 13, anchor="end", color=BLUE, weight="bold")
    for ang, col, lab in ((60, ORANGE, "inferred channel"), (84, GREY, "random channel")):
        a = math.radians(ang)
        ex, ey = ox + L * math.cos(a), oy - L * math.sin(a)
        S.arrow(ox, oy, ex, ey, col, sw=3)
        S.line(ex, ey, ex, oy, col, 1.2, dashed=True)
        S.text(ex + 8, ey - 6, lab, 13, anchor="start", color=col, weight="bold")
    # bars
    S.text(560, 26, "Authority (cos² of the angle)", 15, weight="bold")
    bars = [("best channel (0°)", 1.0, BLUE), ("inferred (60°)", 0.25, ORANGE), ("random (84°)", 0.011, GREY)]
    for i, (lab, v, c) in enumerate(bars):
        y = 60 + i * 52
        S.text(560, y + 18, lab, 13, anchor="end")
        S.add(f'<rect x="570" y="{y}" width="{max(2, 150 * v)}" height="26" fill="{c}"/>')
        S.text(578 + 150 * v, y + 18, f"{v*100:.0f}%", 13, anchor="start")
    # error-removed curve
    S.text(560, 236, "LQR then removes  G / (G + ρ)  of the error", 14, weight="bold")
    S.text(560, 258, "ρ = effort cost, set so driving every unit removes 90%", 13, color=MUTE)
    S.text(560, 284, "best: 90%   inferred: 69%   random: 8%", 14)
    S.text(560, 306, "(the teacher's numbers in the load window)", 12, color=MUTE)
    S.save("d7_authority")


# ---------------------------------------------------------------- D8: mean difference vs push
def d8():
    S = SVG(900, 300, "A mean-difference axis is a chord across the curved manifold, not the direction to push")
    cx, cy, R = 300, 360, 270
    pts = [(cx + R * math.cos(math.radians(a)), cy - R * math.sin(math.radians(a))) for a in range(150, 29, -5)]
    S.path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts), GRID, 12)
    S.path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts), MUTE, 2)
    lo = [(cx + R * math.cos(math.radians(a)), cy - R * math.sin(math.radians(a))) for a in (140, 135, 130)]
    hi = [(cx + R * math.cos(math.radians(a)), cy - R * math.sin(math.radians(a))) for a in (50, 45, 40)]
    for x, y in lo: S.circle(x, y, 7, "#9ec5f4", INK, 1)
    for x, y in hi: S.circle(x, y, 7, "#0d366b", INK, 1)
    ml = (sum(p[0] for p in lo) / 3, sum(p[1] for p in lo) / 3 + 18)
    mh = (sum(p[0] for p in hi) / 3, sum(p[1] for p in hi) / 3 + 18)
    S.text(ml[0] - 30, ml[1] + 62, "low-plan trials", 13, color=MUTE)
    S.text(mh[0] + 30, mh[1] + 62, "high-plan trials", 13, color=MUTE)
    S.arrow(ml[0], ml[1], mh[0] - 4, mh[1], ORANGE, sw=3)
    S.text(cx, ml[1] + 24, "mean-difference axis (SURF)", 13, color=ORANGE, weight="bold")
    tx, ty = cx, cy - R
    S.arrow(tx - 110, ty, tx + 110, ty, MUTE, sw=2)
    S.text(tx + 116, ty + 5, "local tangent", 13, anchor="start", color=MUTE)
    a = math.radians(-28)
    S.arrow(tx, ty, tx + 130 * math.cos(a), ty - 130 * math.sin(a), BLUE, sw=3)
    S.text(tx + 122, ty + 52, "best push", 13, anchor="start", color=BLUE, weight="bold")
    S.circle(tx, ty, 5, INK)
    S.text(660, 70, "On the teacher:", 15, anchor="start", weight="bold")
    for i, ln in enumerate(["chord vs tangent: 42°", "chord vs best push: 48–52°", "chord in slow subspace: 0.36",
                            "(SURF: 0.352)", "", "yet one channel along the best", "push steers it completely"]):
        S.text(660, 98 + i * 22, ln, 14, anchor="start", color=MUTE if ln.startswith("(") else INK)
    S.save("d8_meandiff")


# ---------------------------------------------------------------- D9: non-normal push
def d9():
    S = SVG(900, 320, "With slanted (non-normal) dynamics the best push is not along the line")
    for k, (title, slant) in enumerate((("Ordinary case", 90), ("Slanted case (the teacher)", 129))):
        ox, oy = 90 + k * 450, 240
        S.text(ox + 160, 28, title, 15, weight="bold")
        S.arrow(ox - 20, oy, ox + 330, oy, MUTE, sw=2)
        S.text(ox + 330, oy + 24, "slow direction (the line)", 13, anchor="end", color=MUTE)
        fa = math.radians(slant)
        S.arrow(ox + 150, oy, ox + 150 + 150 * math.cos(fa), oy - 150 * math.sin(fa), RED, sw=2, dashed=True)
        S.text(ox + 150 + 150 * math.cos(fa) - 6, oy - 150 * math.sin(fa) - 8, "fast direction (decays)", 12,
               anchor="middle", color=RED)
        L = 120
        # push along the line
        S.arrow(ox, oy, ox + L, oy, ORANGE, sw=4)
        if k == 0:
            S.text(ox + 60, oy + 44, "push along the line: whole push is kept", 13, anchor="start")
        else:
            la = math.radians(39)
            ex, ey = ox + L * math.cos(la), oy - L * math.sin(la)
            S.arrow(ox, oy, ex, ey, BLUE, sw=4)
            S.text(ex - 6, ey - 10, "best push (39° off)", 13, anchor="middle", color=BLUE, weight="bold")
            # project along fast direction onto the line
            t = (oy - ey) / math.sin(fa)
            px = ex - t * math.cos(fa)
            S.line(ex, ey, px, oy, RED, 1.5, dashed=True)
            S.circle(px, oy, 5, BLUE)
            S.text(ox + 40, oy + 44, "same size push, after the fast part decays", 13, anchor="start")
            S.text(ox + 40, oy + 62, f"it leaves a {(px-ox)/L:.2f}× bigger lasting shift (blue dot)", 13, anchor="start")
    S.save("d9_nonnormal")


# ---------------------------------------------------------------- D10: two goals
def d10():
    S = SVG(900, 320, "Moving one behavioural number is much easier than placing the whole state")
    # left: contour lines of equal pull
    S.text(215, 26, "Goal A: change the pull", 15, weight="bold")
    for i in range(5):
        x = 80 + i * 60
        S.line(x - 40, 270, x + 40, 50, BLUE if i == 3 else GRID, 4 if i == 3 else 1.5)
    S.text(300, 300, "target pull: any state on the blue line", 13, color=BLUE)
    S.text(215, 48, "(grey lines: states with equal pull)", 12, color=MUTE)
    S.circle(130, 175, 8, INK); S.text(110, 160, "now", 13, anchor="end")
    a = math.atan2(-60, 220)
    S.arrow(138, 172, 252, 140, ORANGE, sw=3)
    S.text(170, 205, "one channel, aimed", 13, color=ORANGE, weight="bold")
    S.text(215, 318, "", 1)
    # right: target point
    S.text(670, 26, "Goal B: put the whole state at a point", 15, weight="bold")
    S.circle(560, 220, 8, INK); S.text(545, 245, "now", 13, anchor="end")
    S.add(f'<circle cx="760" cy="90" r="9" fill="none" stroke="{RED}" stroke-width="3"/>')
    S.line(752, 82, 768, 98, RED, 2); S.line(752, 98, 768, 82, RED, 2)
    S.text(790, 85, "target state", 13, anchor="start", color=RED)
    S.arrow(568, 216, 720, 180, ORANGE, sw=3)
    S.path("M720,180 L760,90", MUTE, 1.5, dashed=True)
    S.text(600, 172, "one channel gets part way", 13, color=ORANGE)
    S.text(670, 280, "needs a channel for every direction:", 13)
    S.text(670, 298, "4 channels reach a 4-D slice of 128-D", 13)
    S.save("d10_goals")


# ---------------------------------------------------------------- D11: verdict flowchart
def d11():
    S = SVG(900, 470, "Each explanation, the test used, the result and the verdict")
    S.box(270, 10, 360, 44, None, "Why did steering remove only 0.1%?", fill=LGREY, stroke=INK, size=15)
    hdr = [(20, "Explanation"), (275, "Test"), (525, "Result"), (790, "Verdict")]
    for x, h in hdr:
        S.text(842 if h == "Verdict" else x + 110, 80, h, 14, weight="bold", color=MUTE)
    rows = [("Too few channels", "best single channel\non the teacher", "100% of full authority;\nmonkey model: best 4 = 84–98%", "No", LGREY, MUTE),
            ("Channels aimed wrong", "inferred vs best channels,\nseveral seeds", "students 15–51% vs 56–100%;\nmonkey 4–18% vs 84–98%", "Yes", LGREEN, "#11795a"),
            ("Real inputs miss the plan", "teacher's own input\nchannels in the delay", "only 15%, but still removes\n57% of the error", "Partly", LORANGE, ORANGE),
            ("Goal was the whole state", "same channels, whole-state\ngoal vs behaviour goal", "whole-state goal scores real,\nrandom and best all ~13–14%", "Maybe", LBLUE, BLUE),
            ("Effort cost set too high", "sweep the effort cost\non the teacher", "0.1% needs a cost\n300–10,000× the fair value", "Likely", LBLUE, BLUE)]
    for i, (h, t, r, v, f, c) in enumerate(rows):
        y = 95 + i * 72
        S.box(20, y, 220, 60, None, h, fill="#ffffff", stroke=INK, size=14)
        S.box(275, y, 220, 60, None, t, fill="#ffffff", stroke=GRID, size=13)
        S.box(525, y, 245, 60, None, r, fill="#ffffff", stroke=GRID, size=13)
        S.arrow(240, y + 30, 272, y + 30, MUTE); S.arrow(495, y + 30, 522, y + 30, MUTE); S.arrow(770, y + 30, 797, y + 30, MUTE)
        chip(S, 800, y + 15, 85, 30, v, f, c)
    S.save("d11_verdict")


for f in (d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11):
    f()
print("ok", sorted(os.listdir(OUT)))


# ---------------------------------------------------------------- D12: a line inside a high-dimensional space
def d12():
    S = SVG(900, 360, "The plan lives on a thin curved thread inside the 100-dimensional state space")
    # box in perspective
    bx, by, bw, bh, dx, dy = 40, 70, 380, 230, 70, -45
    S.add(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="none" stroke="{GRID}" stroke-width="1.5"/>')
    S.add(f'<rect x="{bx+dx}" y="{by+dy}" width="{bw}" height="{bh}" fill="none" stroke="{GRID}" stroke-width="1.5"/>')
    for (x1, y1) in ((bx, by), (bx + bw, by), (bx, by + bh), (bx + bw, by + bh)):
        S.line(x1, y1, x1 + dx, y1 + dy, GRID, 1.5)
    S.text(255, 22, "State space: one axis per unit (100 for the teacher)", 14, weight="bold")
    S.text(255, 345, "drawn in 3-D, the real space has 100 axes", 12, color=MUTE)
    # thread
    pts = [(120 + i * 42, 230 - 95 * math.sin(math.pi * i / 7) + i * 4) for i in range(8)]
    S.path("M" + " Q".join([f"{pts[0][0]},{pts[0][1]}"] + [f"{(pts[i][0]+pts[i+1][0])/2},{(pts[i][1]+pts[i+1][1])/2} {pts[i+1][0]},{pts[i+1][1]}" for i in range(0, 7, 2)]) if False else
           "M" + " L".join(f"{x:.0f},{y:.0f}" for x, y in pts), "#bcd6f5", 9)
    cols = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b", "#0a2a52"]
    for (x, y), c in zip(pts, cols):
        S.circle(x, y, 6, c, INK, 0.8)
    S.text(pts[0][0] - 8, pts[0][1] + 26, "weak plan", 12, color=MUTE)
    S.text(pts[-1][0] + 10, pts[-1][1] + 26, "strong plan", 12, color=MUTE)
    # trajectories converging onto the thread
    for (sx, sy), (tx, ty) in (((90, 120), pts[1]), ((400, 110), pts[5]), ((330, 290), pts[4]), ((150, 300), pts[2])):
        S.path(f"M{sx},{sy} Q{(sx+tx)/2+25},{(sy+ty)/2-20} {tx},{ty}", ORANGE, 1.8, dashed=True, arrow=True)
        S.circle(sx, sy, 4, ORANGE)
    S.text(95, 105, "start of trial", 12, color=ORANGE)
    # right: explanation of decay
    S.text(650, 60, "Off the thread: pulled back", 14, weight="bold")
    vals = [1.0, 0.64, 0.41, 0.26, 0.17, 0.11]
    for i, v in enumerate(vals):
        x = 540 + i * 40
        S.add(f'<rect x="{x}" y="{190 - 110*v:.0f}" width="26" height="{110*v:.0f}" fill="{RED}" opacity="0.8"/>')
        S.text(x + 13, 208, str(i), 12, color=MUTE)
        S.text(x + 13, 184 - 110 * v, f"{v:.2f}", 11, color=MUTE)
    S.text(650, 228, "steps after the push", 12, color=MUTE)
    S.text(650, 252, "each step keeps 64% of a push off the thread", 13)
    S.text(650, 285, "Along the thread: stays", 14, weight="bold")
    for i in range(6):
        x = 540 + i * 40
        S.add(f'<rect x="{x}" y="300" width="26" height="22" fill="{BLUE}" opacity="0.8"/>')
    S.text(650, 342, "each step keeps ~100% (eigenvalue ≈ 1): memory", 13)
    S.save("d12_thread")


# ---------------------------------------------------------------- D13: decoder ambiguity
def d13():
    S = SVG(900, 300, "When states lie on a thin line, many different decoders predict equally well")
    ox, oy = 90, 250
    S.line(ox, oy, ox + 330, oy, MUTE, 1.5); S.line(ox, oy, ox, oy - 220, MUTE, 1.5)
    S.text(ox + 330, oy + 20, "unit 1", 12, anchor="end", color=MUTE)
    S.text(ox - 10, oy - 210, "unit 2", 12, anchor="end", color=MUTE)
    for i in range(9):
        x = ox + 40 + i * 30; y = oy - 30 - i * 20
        S.circle(x, y, 7, ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#3987e5", "#256abf", "#184f95", "#0d366b", "#0a2a52"][i], INK, 0.8)
    S.text(ox + 190, 22, "Trials lie on a line (unit 2 rises with unit 1)", 14, weight="bold")
    S.text(ox + 290, oy - 210, "stronger pull", 12, color=MUTE)
    S.text(620, 70, "Rule A: pull = unit 1 only", 14, anchor="start", color=ORANGE, weight="bold")
    S.text(620, 95, "Rule B: pull = unit 2 only", 14, anchor="start", color=GREEN, weight="bold")
    S.text(620, 120, "Rule C: pull = a mix of both", 14, anchor="start", color=BLUE, weight="bold")
    S.text(620, 160, "All three predict every trial", 13, anchor="start")
    S.text(620, 180, "perfectly, but point in very", 13, anchor="start")
    S.text(620, 200, "different directions.", 13, anchor="start")
    S.text(620, 235, "The fitting setting (ridge)", 13, anchor="start", color=MUTE)
    S.text(620, 255, "decides which one you get.", 13, anchor="start", color=MUTE)
    S.arrow(ox + 160, oy - 110, ox + 260, oy - 110, ORANGE, sw=3)
    S.arrow(ox + 160, oy - 110, ox + 160, oy - 200, GREEN, sw=3)
    S.arrow(ox + 160, oy - 110, ox + 240, oy - 175, BLUE, sw=3)
    S.save("d13_decoders")


d12(); d13()
