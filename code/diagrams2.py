"""New schematics for the final report: updated verdict, the swap test, the 2x2 of truth against assumption."""
import diagrams as D
from diagrams import SVG, chip, INK, MUTE, GRID, BLUE, ORANGE, GREEN, GREY, RED, LBLUE, LORANGE, LGREEN, LGREY, LRED


def verdict():
    S = SVG(900, 617, "Each explanation, the test used, the result and the verdict")
    S.box(270, 10, 360, 44, None, "Why did steering remove only 0.1%?", fill=LGREY, stroke=INK, size=15)
    for x, h in [(20, "Explanation"), (275, "Test"), (525, "Result"), (790, "Verdict")]:
        S.text(842 if h == "Verdict" else x + 110, 80, h, 14, weight="bold", color=MUTE)
    rows = [("Too few channels", "best single channel\non the teacher", "100% of full authority;\nmonkey models: best 4 = 84–100%", "No", LGREY, MUTE),
            ("Channels aimed wrong", "inferred vs best channels,\nseveral seeds", "simplified 15–51% vs 56–100%;\nmonkey 4–18% vs 84–98%", "Yes", LGREEN, "#11795a"),
            ("Gate shuts the input out\nof the memory (LFADS)", "full LFADS: inferred vs\nrandom, update gates", "0.04–0.8% of full, below\nrandom (6%); MC_Maze 0.3–0.5%", "Yes", LGREEN, "#11795a"),
            ("Model does not hold the\nplan as the circuit does", "swap, shift, persistence,\nplan before the cue", "simplified: own share 0.03 (truth 1);\nLFADS: plan known before the cue", "Yes", LGREEN, "#11795a"),
            ("Real inputs miss the plan", "teacher's own input\nchannels in the delay", "only 15%, but still removes\n57% of the error", "Partly", LORANGE, ORANGE),
            ("Goal was the whole state", "same channels, whole-state\ngoal vs behaviour goal", "whole-state goal scores real,\nrandom and best all ~13–14%", "Maybe", LBLUE, BLUE),
            ("Effort cost set too high", "sweep the effort cost", "simplified: needs 300–10,000×;\nfull LFADS: not needed", "Possible", LBLUE, BLUE)]
    for i, (h, t, r, v, f, c) in enumerate(rows):
        y = 95 + i * 72
        S.box(20, y, 220, 60, None, h, fill="#ffffff", stroke=INK, size=14)
        S.box(275, y, 220, 60, None, t, fill="#ffffff", stroke=GRID, size=13)
        S.box(525, y, 245, 60, None, r, fill="#ffffff", stroke=GRID, size=13)
        S.arrow(240, y + 30, 272, y + 30, MUTE); S.arrow(495, y + 30, 522, y + 30, MUTE); S.arrow(770, y + 30, 797, y + 30, MUTE)
        chip(S, 800, y + 15, 85, 30, v, f, c)
    S.save("d11_verdict")


def swap():
    S = SVG(900, 400, "The swap test")
    S.text(450, 26, "The swap test: where is the plan kept?", 17, weight="bold")
    # timeline
    def timeline(y, name, col, plan, fill):
        S.text(20, y + 5, name, 14, anchor="start", weight="bold", color=col)
        S.line(150, y, 560, y, MUTE, 1.5)
        S.box(170, y - 16, 60, 32, None, plan, fill=fill, stroke=col, size=13)
        S.box(470, y - 16, 44, 32, None, "go", fill=LGREY, stroke=MUTE, size=13)
    timeline(80, "trial i", BLUE, "plan 0.9", LBLUE)
    timeline(150, "trial j", ORANGE, "plan 0.3", LORANGE)
    S.line(262, 55, 262, 180, RED, 2, dashed=True)
    S.text(262, 200, "swap point", 12, color=RED)
    S.path("M 275 150 C 330 150, 330 80, 385 80", ORANGE, 2.5, arrow=True)
    S.text(395, 118, "from here on, trial i gets\ntrial j's inputs", 12, anchor="start", color=ORANGE)
    S.text(150, 48, "state and inputs of trial i kept", 12, anchor="start", color=BLUE)
    # outcomes
    S.box(600, 50, 280, 62, "Pull comes out as 0.9", "the plan was in the state\n(internal memory)", fill=LBLUE, stroke=BLUE, size=13)
    S.box(600, 128, 280, 62, "Pull comes out as 0.3", "the plan was in the inputs", fill=LORANGE, stroke=ORANGE, size=13)
    S.arrow(565, 80, 597, 80, BLUE); S.arrow(565, 150, 597, 158, ORANGE)
    # results strip
    S.text(450, 240, "What happened", 14, weight="bold", color=MUTE)
    cells = [("Teacher A\n(true attractor)", "follows its own plan\nown share 1.00", LBLUE, BLUE),
             ("Teacher B\n(outside input)", "follows the swapped input\nown share 0.00", LORANGE, ORANGE),
             ("Free students of A", "follow the swapped input\nown share about 0.03", LORANGE, ORANGE),
             ("Free students of B", "follow the swapped input\nown share about 0.02", LORANGE, ORANGE)]
    for i, (a, b, f, c) in enumerate(cells):
        x = 20 + i * 220
        S.box(x, 258, 200, 50, None, a, fill="#ffffff", stroke=INK, size=13)
        S.box(x, 316, 200, 56, None, b, fill=f, stroke=c, size=13)
    S.save("d14_swap")


def grid():
    S = SVG(900, 420, "Truth against the model's assumption")
    S.text(450, 26, "What the model reports follows its assumption, not the truth", 17, weight="bold")
    S.text(395, 62, "Free student\n(input allowed at any time)", 13, weight="bold", color=MUTE)
    S.text(705, 62, "Told-when student\n(input only at the two cues)", 13, weight="bold", color=MUTE)
    rows = [("Teacher A", "plan held by an\ninternal attractor", BLUE,
             ("reports: plan supplied by inputs", "WRONG", "held-out 0.22–0.29", LRED, RED),
             ("reports: one slow direction\nholds the plan", "RIGHT", "held-out 0.34–0.37", LGREEN, "#11795a")),
            ("Teacher B", "plan held by an\noutside input", ORANGE,
             ("reports: plan supplied by inputs", "RIGHT", "held-out 0.09–0.25", LGREEN, "#11795a"),
             ("reports: a slow direction holds\nthe plan (2 of 6 fits; 4 failed)", "WRONG", "held-out 0.15–0.31", LRED, RED))]
    for i, (name, sub, col, c1, c2) in enumerate(rows):
        y = 105 + i * 150
        S.box(20, y, 200, 120, name, sub, fill="#ffffff", stroke=col, size=13, tcolor=col)
        for j, (body, verdict, ho, f, c) in enumerate((c1, c2)):
            x = 250 + j * 310
            S.box(x, y, 290, 120, None, None, fill=f, stroke=c)
            S.text(x + 145, y + 30, body, 13)
            chip(S, x + 20, y + 76, 90, 28, verdict, "#ffffff", c)
            S.text(x + 200, y + 95, ho, 12, color=MUTE)
    S.text(450, 408, "Of these two models, the best held-out prediction (bits per spike) on each teacher came from a told-when fit, including on teacher B, where it is wrong.", 12, color=MUTE)
    S.save("d15_grid")


verdict(); swap(); grid()
