"""Transfer test for simplified students fitted to the noisy teacher (dynamics noise 0.1),
with the teacher's true sensitivity taken along its actual noisy path."""
import glob, json
import numpy as np, torch
import steer as S
from teacher import load_teacher
from student_steer import load_student, lever_decoder
from event_tests import bridge
from lfads_tests import teacher_sens
torch.set_num_threads(1)
out = {}
for dp, pat in (("runs/dataset_An.pt", "runs/student_k4_lam0.1_s?_An.pt"), ("runs/dataset.pt", "runs/student_k4_lam0.1_s[01].pt")):
    data = torch.load(dp, weights_only=False); tl, tg = data["t_load"], data["t_go"]
    net, _ = load_teacher("runs/teacher_g0.9_n0.05_s0.pt")
    sT = teacher_sens(net, data, tg); idx = S.window_idx("delay", tl, tg)
    for p in sorted(glob.glob(pat)):
        st, ck = load_student(p, data)
        w, b, r2 = lever_decoder(st, data, 1e-2)
        sS, _ = S.student_sensitivity(st, w, b, tg)
        with torch.no_grad():
            G, _ = st.generate(use_u=True)
        DF = (st.Dout.detach() @ st.F.detach()).numpy()
        br = bridge(data, net, sT, sS, G.numpy(), DF, list(range(64)), idx)
        out[p] = br; print(p, {k: round(v, 3) for k, v in br.items()}, flush=True)
json.dump(out, open("runs/noisy_bridge.json", "w"), indent=1)
