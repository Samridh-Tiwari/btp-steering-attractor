"""Spikes from teacher A and teacher A2 (the second seed): runs/dataset.pt and runs/dataset_T2.pt.
Teacher B's data come from split_study.py data_b, the noisy teacher's from make_noisy.py."""
import torch
from teacher import load_teacher
from lfads_lite import make_dataset

for path, out in (("runs/teacher_g0.9_n0.05_s0.pt", "runs/dataset.pt"),
                  ("runs/teacher_g0.9_n0.05_s1.pt", "runs/dataset_T2.pt")):
    net, _ = load_teacher(path)
    torch.save(make_dataset(net), out)
    print("saved", out)
