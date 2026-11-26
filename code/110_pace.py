"""PACE (Yu, Wang, Liao & Wu, NeurIPS 2026; github.com/AI4Science-WestlakeU/PACE, commit 5f5a10e, MIT)
under this analysis' protocol.

PACE is trained with its own entry point (train.py, Hydra) and its preset for scRNA-seq PCA data
(experiment=op_cite_pace_ode_100d: Sinkhorn-matched geometry-aware bridges, 120 Stage-1 epochs with
rematching every 20, then a 1,000-epoch ODE velocity field; whitening fitted on the training days),
on the same 30-component PCA as every other method (an AnnData with obsm["X_pca"] and obs["day"],
read by PACE's own CITE-seq loader). Only the dimension, the data path and the time labels are
overridden. PACE maps the training labels to [0, 1] by (label - min) / (max - min) and integrates
its velocity field with Euler steps (src/inference.py); the training-only StandardScaler is refitted
here on the same training cells, exactly as its data module does.

Fate: each early cell is integrated from the first to the last day (deterministic ODE, one
simulation) and classified by its 20 nearest annotated last-day cells (110_common.py). Fit: first-day
cells integrated to each later day, W2 against real cells (as 100). Held-out: trained on the first and
last days only, in 106's PCA of the end days; first-day cells integrated to the middle day.

Usage: 110_pace.py fate <h5ad> <featdir> <dataset> <workdir>
       110_pace.py heldout <h5ad> <featdir> <dataset> <workdir> <out.csv> [--seeds 3]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PACE = os.path.abspath(os.path.join(HERE, "..", "ext", "pace_repo", "src"))
PY_PACE = os.path.abspath(os.path.join(HERE, "..", "ext", "env_pace", "bin", "python"))
spec = importlib.util.spec_from_file_location("c110", os.path.join(HERE, "110_common.py"))
c110 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c110)
say = c110.say

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["fate", "heldout"])
ap.add_argument("h5ad")
ap.add_argument("featdir")
ap.add_argument("dataset")
ap.add_argument("workdir")
ap.add_argument("out", nargs="?", default="")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--seed-start", type=int, default=0)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--extra", default="", help="extra Hydra overrides (smoke tests only)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--experiment", default="op_cite_pace_ode_100d",
                help="PACE preset; documented scRNA-PCA presets: op_cite_pace_ode_100d, ipsc_liu_pace_ode_50d")
ap.add_argument("--name", default="snap_pace", help="output name (fate) / method label (held-out)")
a = ap.parse_args()
a.workdir = os.path.abspath(a.workdir)


def run_pace(Z, day, train_days, tag, seed):
    """Train PACE on cells of train_days; returns the run directory."""
    wd = os.path.join(a.workdir, tag)
    os.makedirs(wd, exist_ok=True)
    m = np.isin(day, train_days)
    A = ad.AnnData(X=np.zeros((int(m.sum()), 1), dtype=np.float32))
    A.obsm["X_pca"] = Z[m].astype(np.float32)
    A.obs["day"] = day[m].astype(float)
    h5 = os.path.join(wd, "input.h5ad")
    A.write_h5ad(h5)
    labels = "[" + ",".join("%g" % d for d in train_days) + "]"
    name = "cite_" + a.dataset  # PACE's CITE-seq loader reads obsm["X_pca"] and obs["day"]
    cmd = [PY_PACE, "train.py", "experiment=" + a.experiment, "data_name=" + name, "time_column=day", "feature_key=X_pca", "data_path=" + h5,
           "dim=%d" % Z.shape[1], "train_timepoint_labels=" + labels, "test_timepoint_labels=[]",
           "working_dir=" + wd, "seed=%d" % seed, "seeds=[%d]" % seed, "enable_progress_bar=false",
           "hydra.run.dir=" + os.path.join(wd, "hydra")] + [x for x in a.extra.split(" ") if x]
    say("PACE: " + " ".join(cmd[1:]))
    t0 = time.time()
    with open(os.path.join(wd, "train.log"), "w") as fh:
        r = subprocess.run(cmd, cwd=PACE, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError("PACE training failed (see %s/train.log)" % wd)
    say("trained in %.0f s" % (time.time() - t0))
    return os.path.join(wd, "results", "%s_dim%d" % (name, Z.shape[1]), "pace"), Z[m]


class Flow:
    """PACE's trained velocity field with its training-only scaler and time map."""

    def __init__(self, run_dir, train_Z, train_days):
        sys.path.insert(0, PACE)
        from src.inference import load_run, rollout  # noqa: E402
        from sklearn.preprocessing import StandardScaler
        self.model, self.cfg, self.ckpt = load_run(run_dir, device=a.device)
        self.rollout = rollout
        self.sc = StandardScaler().fit(train_Z)
        lo, hi = min(train_days), max(train_days)
        self.t = lambda d: (d - lo) / (hi - lo)

    def to_day(self, X0, d_from, d_to, steps=101):
        x = self.sc.transform(X0).astype(np.float32)
        out = []
        for i in range(0, len(x), 4096):
            tr = self.rollout(self.model, x[i:i + 4096], t_start=self.t(d_from), t_end=self.t(d_to),
                              steps=steps, device=a.device)
            out.append(tr[-1].numpy())
        return self.sc.inverse_transform(np.concatenate(out))


def fate_mode():
    F = c110.FateData(a.h5ad, a.featdir)
    days = F.days
    run_dir, trZ = run_pace(F.Z, F.day, days, "fate_" + a.experiment, seed=42)
    fl = Flow(run_dir, trZ, days)
    fit_ref = F.fit_reference(c110.w2)
    fit = {"sim_day%g_vs_day%g" % (d, d): c110.w2(fl.to_day(F.fit_start(), days[0], d), F.fit_target(d))
           for d in days[1:]}
    say("fit references %s" % {k: round(v, 3) for k, v in fit_ref.items()})
    say("fit of PACE %s" % {k: round(v, 3) for k, v in fit.items()})
    E = fl.to_day(F.Z[F.d2i], days[0], days[-1])
    P = F.classify(E)
    F.save(a.featdir, a.name, P, dict(method="PACE", commit="5f5a10ed6c7601fdab017ced51f7e797557b182b",
                                          preset=a.experiment, checkpoint=str(fl.ckpt),
                                          fit_reference=fit_ref, fit=fit, n_early=int(len(E))))
    say("mean probability %s" % dict(zip(F.pops, P.mean(0).round(3))))


def heldout_mode():
    hc = c110.heldout_common()
    B, X, day, days = hc.build(a.h5ad, a.featdir)
    pca, Zraw = hc.space(X, day, days)
    rows = []
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        S = hc.draw(day, days, seed, a.n)
        t0 = time.time()
        run_dir, trZ = run_pace(Zraw, day, [days[0], days[2]], "heldout_%s_s%d" % (a.experiment, seed), seed=9400 + seed)
        fl = Flow(run_dir, trZ, [days[0], days[2]])
        P = fl.to_day(Zraw[S["first"]], days[0], days[1])
        d = hc.w2(P, Zraw[S["eval"]])
        rows.append(dict(dataset=a.dataset, method=a.name.replace("snap_", ""), seed=seed, w2=d, n_eval=len(S["eval"]),
                         seconds=time.time() - t0))
        say("%s seed %d: W2 to the held-out day %.3f" % (a.name, seed, d))
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    say("PACE %s %s %s" % (a.mode, a.dataset, a.featdir))
    fate_mode() if a.mode == "fate" else heldout_mode()
    say("done")
