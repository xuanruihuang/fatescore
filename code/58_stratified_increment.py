"""The stacked increment within the clones the expression reference finds hard.

A reviewer's objection: Neutrophil versus Monocyte is largely written in day-2
expression, so an increment measured over all clones is dominated by clones
whose fate the transcriptome already reads, and a method could help exactly
where the transcriptome is unsure without it showing. This splits the clones by
how sure the reference is -- the distance of its repeat-averaged out-of-fold
score from 0.5 (the reference here is the stacked base, ST_base) -- into the
less sure half and the surer half, and recomputes each method's stacked
increment within each half: mean over repeats of the AUROC difference, and the
repeat-averaged paired bootstrap of 56_stacked_increment.py, widened by the
partition term. The split uses only out-of-fold scores, never the labels.

Usage: 58_stratified_increment.py <stk_prefix> [<stk_prefix> ...]
reads <prefix>_oof.npz, writes <prefix>_strata.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOT = 2000
SEED = 13


def auroc_rows(score, y):
    B, n = score.shape
    n1 = y.sum(1).astype(np.float64)
    n0 = n - n1
    order = np.argsort(score, axis=1, kind="stable")
    ranks = np.empty((B, n), dtype=np.float64)
    seq = np.broadcast_to(np.arange(1, n + 1, dtype=np.float64), (B, n))
    np.put_along_axis(ranks, order, seq, axis=1)
    out = ((ranks * y).sum(1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    out[(n1 == 0) | (n0 == 0)] = np.nan
    return out


for pre in sys.argv[1:]:
    Z = np.load(pre + "_oof.npz", allow_pickle=True)
    keys = list(Z["keys"])
    oof, y, g = Z["oof"], Z["y"], Z["groups"]
    R = int(Z["repeats"])
    clones = np.unique(g)
    # clone-level scores: (arm, repeat, clone)
    inv = np.searchsorted(clones, g)
    cnt = np.bincount(inv, minlength=clones.size)
    S = np.stack([[np.bincount(inv, weights=oof[i, r], minlength=clones.size)
                   / cnt for r in range(R)] for i in range(len(keys))])
    yc = np.zeros(clones.size, bool)
    yc[inv] = y.astype(bool)
    base = S[keys.index("ST_base")].mean(0)
    sure = np.abs(base - 0.5)
    hard = sure <= np.median(sure)
    rows = []
    for name, mask in (("all", np.ones_like(hard)), ("less_sure_half", hard),
                       ("surer_half", ~hard)):
        idx_c = np.where(mask)[0]
        yy = yc[idx_c]
        rng = np.random.default_rng(SEED)
        bidx = rng.integers(0, idx_c.size, size=(N_BOOT, idx_c.size))
        yb = yy[bidx]
        ref = keys.index("ST_base")
        bref = [auroc_rows(S[ref, r, idx_c][bidx], yb) for r in range(R)]
        aref = [roc_auc_score(yy, S[ref, r, idx_c]) for r in range(R)]
        for k in keys:
            if not k.startswith("ST_") or k == "ST_base":
                continue
            i = keys.index(k)
            d = np.array([roc_auc_score(yy, S[i, r, idx_c]) - aref[r]
                          for r in range(R)])
            davg = np.mean([auroc_rows(S[i, r, idx_c][bidx], yb) - bref[r]
                            for r in range(R)], axis=0)
            davg = davg[np.isfinite(davg)]
            q = np.percentile(davg, [2.5, 97.5])
            m, sd = float(d.mean()), float(d.std(ddof=1))
            part = 1.96 * sd / np.sqrt(R)
            rows.append(dict(stratum=name, arm=k[3:], n_clones=int(idx_c.size),
                             prevalence=round(float(yy.mean()), 3),
                             ref_auroc=round(float(np.mean(aref)), 4),
                             mean10=round(m, 4), sd10=round(sd, 4),
                             lo=round(m - np.sqrt((m - q[0]) ** 2 + part ** 2), 4),
                             hi=round(m + np.sqrt((q[1] - m) ** 2 + part ** 2), 4)))
    D = pd.DataFrame(rows)
    D.to_csv(pre + "_strata.csv", index=False)
    print(pre)
    print(D.to_string(index=False))
