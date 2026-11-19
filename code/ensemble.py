"""
Do different fits to the same spikes agree? (Malonis et al. 2026 ask the same of LFADS
ensembles on real data.)  For every pair of fits of one kind to one teacher:

  state agreement : cross-validated R^2 of a linear map from fit i's generator state to
                    fit j's (fit on even trials, scored on odd), averaged both ways
  input agreement : the same for the inferred inputs, with lags -2..+2 allowed
  truth recovery  : R^2 of the teacher's real inputs from each fit's inferred inputs
                    (lags -2..+2), cross-validated the same way

    python ensemble.py
"""
import glob
import json
import re
import itertools

import numpy as np
import torch

from student_steer import load_student


def lagged(X, lags=(-2, -1, 0, 1, 2)):
    n, T, d = X.shape
    out = np.zeros((n, T, d * len(lags)))
    for j, l in enumerate(lags):
        sh = np.roll(X, l, axis=1)
        if l > 0:
            sh[:, :l] = 0
        elif l < 0:
            sh[:, l:] = 0
        out[..., j * d:(j + 1) * d] = sh
    return out


def cv_r2(X, Y, ridge=1e-3):
    """X (n,T,p) -> Y (n,T,q); fit on even trials, score on odd. Variance-weighted R^2."""
    tr = np.arange(X.shape[0]) % 2 == 0
    Xa = X[tr].reshape(-1, X.shape[-1]); Ya = Y[tr].reshape(-1, Y.shape[-1])
    Xb = X[~tr].reshape(-1, X.shape[-1]); Yb = Y[~tr].reshape(-1, Y.shape[-1])
    mx, my = Xa.mean(0), Ya.mean(0)
    Xc = Xa - mx
    sc = np.trace(Xc.T @ Xc) / Xc.shape[1]
    Wt = np.linalg.solve(Xc.T @ Xc + ridge * sc * np.eye(Xc.shape[1]), Xc.T @ (Ya - my))
    P = (Xb - mx) @ Wt + my
    return float(1 - ((P - Yb) ** 2).sum() / ((Yb - Yb.mean(0)) ** 2).sum())


def pairwise(fits, key, lags):
    vals = []
    for a, b in itertools.combinations(range(len(fits)), 2):
        Xa = lagged(fits[a][key], lags) if lags else fits[a][key]
        Xb = lagged(fits[b][key], lags) if lags else fits[b][key]
        vals.append(0.5 * (cv_r2(Xa, fits[b][key]) + cv_r2(Xb, fits[a][key])))
    return vals


@torch.no_grad()
def lite_fit(p, data):
    st, ck = load_student(p, data)
    if not np.isfinite(ck["eval"]["bps_heldin"]) or ck["eval"]["bps_heldin"] < 0:
        return None
    G, _ = st.generate(use_u=True)
    return dict(name=p, G=G.numpy(), U=st.U.detach().numpy(), F=(G @ st.F.T).numpy(), R=st.logrates(G).numpy())


@torch.no_grad()
def full_fit(p, data):
    import lfads_tests as LT, lfads_full as L
    m, ck = LT.load(p)
    if "test_idx" in ck["eval"]:
        data = L.subset(data, np.array(ck["eval"]["test_idx"]))
    o = L.infer(m, data)
    return dict(name=p, G=o["gen"].numpy(), U=o["co_mean"].numpy(), F=o["factors"].numpy(), R=o["logr"].numpy())


def summarise(fits, data):
    if len(fits) < 2:
        return None
    lags = (-2, -1, 0, 1, 2)
    st_ag = pairwise(fits, "G", None)
    fa_ag = pairwise(fits, "F", None)
    rt_ag = pairwise(fits, "R", None)
    H = data["H_teacher"]
    st_truth = [cv_r2(f["G"], H) for f in fits]
    in_ag = pairwise(fits, "U", lags)
    u_true = data["u_teacher"]
    truth = [cv_r2(lagged(f["U"], lags), u_true) for f in fits]
    # per channel of the teacher input (plan pulse / go cue, or whatever the teacher has)
    truth_ch = [[cv_r2(lagged(f["U"], lags), u_true[..., c:c + 1]) for c in range(u_true.shape[-1])] for f in fits]
    return dict(n=len(fits), names=[f["name"] for f in fits],
                rates_agree=dict(median=float(np.median(rt_ag)), min=float(np.min(rt_ag)), max=float(np.max(rt_ag))),
                factor_agree=dict(median=float(np.median(fa_ag)), min=float(np.min(fa_ag)), max=float(np.max(fa_ag))),
                teacher_state_from_fit=dict(median=float(np.median(st_truth)), min=float(np.min(st_truth)), max=float(np.max(st_truth))),
                state_agree=dict(median=float(np.median(st_ag)), min=float(np.min(st_ag)), max=float(np.max(st_ag))),
                input_agree=dict(median=float(np.median(in_ag)), min=float(np.min(in_ag)), max=float(np.max(in_ag))),
                truth_from_inputs=dict(median=float(np.median(truth)), min=float(np.min(truth)), max=float(np.max(truth))),
                truth_by_channel=np.median(truth_ch, 0).tolist())


if __name__ == "__main__":
    import sys
    torch.set_num_threads(1)
    which = sys.argv[1] if len(sys.argv) > 1 else "lite"
    out_path = "runs/ensemble.json"
    try:
        out = json.load(open(out_path))
    except Exception:
        out = {}
    sets = {"A": "runs/dataset.pt", "B": "runs/dataset_B.pt", "T2": "runs/dataset_T2.pt"}
    for tag, dp in sets.items():
        data = torch.load(dp, weights_only=False)
        if which == "lite":
            for lam in ("0.1", "0.0001"):
                suf = "" if tag == "A" else "_" + tag
                ps = sorted(p for p in glob.glob(f"runs/student_k4_lam{lam}_s*.pt")
                            if re.match(rf"runs/student_k4_lam{re.escape(lam)}_s\d{suf}\.pt$", p))
                fits = [f for f in (lite_fit(p, data) for p in ps) if f is not None]
                r = summarise(fits, data)
                if r:
                    out[f"lite_{tag}_lam{lam}"] = r
                    print(tag, lam, json.dumps({k: v for k, v in r.items() if k != "names"}), flush=True)
        else:
            ps_all = [p for p in glob.glob(f"runs/lfads_{tag}_kl*_s*_tr192.pt")]
            kls = sorted({re.search(r"_kl([0-9.e-]+)_s", p).group(1) for p in ps_all})
            for kl in kls:
                ps = sorted(p for p in ps_all if f"_kl{kl}_s" in p)
                if len(ps) < 2:
                    continue
                fits = [full_fit(p, data) for p in ps]
                import lfads_full as L
                ck = torch.load(ps[0], weights_only=False)
                r = summarise(fits, L.subset(data, np.array(ck["eval"]["test_idx"])))
                if r:
                    out[f"full_{tag}_kl{kl}"] = r
                    print(tag, kl, json.dumps({k: v for k, v in r.items() if k != "names"}), flush=True)
        json.dump(out, open(out_path, "w"), indent=1)
