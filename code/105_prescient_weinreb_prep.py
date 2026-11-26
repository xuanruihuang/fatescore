"""PRESCIENT's LARRY input from the full Weinreb et al. 2020 in vitro data (GEO GSM4185642, the
files of AllonKleinLab/paper-data): 130,887 cells, total-count-normalised counts.

As PRESCIENT's Methods: highly variable genes as Weinreb et al. (SPRING V-score filter: genes above
the 85th V-score percentile with >= 3 counts in >= 3 cells), then genes correlated with cell-cycle
genes removed (Pearson r >= 0.1 with the summed z-score of Ube2c, Hmgb2, Hmgn2, Tuba1b, Ccnb1,
Tubb5, Top2a, Tubb4b); normalised expression of those genes scaled and projected to 50 principal
components. The V-score and correlation filters are re-implemented from SPRING's helper functions
(get_vscores, filter_genes, remove_corr_genes); SPRING commit a37bbd0 itself is not used.
Growth: none, and KEGG (PRESCIENT's get_growth_weights / compute_growth on KEGG mmu04110 cell cycle
= birth, mmu04210 apoptosis = death, genes among the highly variable genes before the cell-cycle
filter, since the filter removes most cell-cycle genes; smoothing on the 50 components).

Usage: 105_prescient_weinreb_prep.py <weinreb_dir> <kegg_dir> <out_dir>
"""
import gzip
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.io
import scipy.optimize
import scipy.sparse as sp
import sklearn.decomposition
import sklearn.preprocessing
import torch

WD, KD, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
say = lambda m: print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)  # noqa: E731
P = os.path.join(WD, "GSM4185642_stateFate_inVitro_")
npz = os.path.join(OUT, "normed_counts_csr.npz")
if os.path.exists(npz):
    E = sp.load_npz(npz)
else:
    say("reading the normalised counts (mtx.gz)")
    with gzip.open(P + "normed_counts.mtx.gz", "rb") as f:
        E = scipy.io.mmread(f).tocsr().astype(np.float32)
    sp.save_npz(npz, E)
genes = [l.strip() for l in gzip.open(P + "gene_names.txt.gz", "rt")]
meta = pd.read_csv(P + "metadata.txt.gz", sep="\t")
say("matrix %s, genes %d, metadata %s; columns %s" % (E.shape, len(genes), meta.shape, list(meta.columns)))
assert E.shape == (len(meta), len(genes))


def runningquantile(x, y, p, nBins):
    ind = np.argsort(x); x = x[ind]; y = y[ind]
    dx = (x[-1] - x[0]) / nBins
    xOut = np.linspace(x[0] + dx / 2, x[-1] - dx / 2, nBins)
    yOut = np.zeros(xOut.shape)
    for i in range(len(xOut)):
        ind = np.nonzero((x >= xOut[i] - dx / 2) & (x < xOut[i] + dx / 2))[0]
        if len(ind) > 0:
            yOut[i] = np.percentile(y[ind], p)
        else:
            yOut[i] = yOut[i - 1] if i > 0 else np.nan
    return xOut, yOut


def get_vscores(E, min_mean=0, nBins=50, fit_percentile=0.1, error_wt=1):
    mu = np.asarray(E.mean(axis=0)).ravel()
    gix = np.nonzero(mu > min_mean)[0]; mu = mu[gix]
    tmp = E[:, gix].copy(); tmp.data **= 2
    var = np.asarray(tmp.mean(axis=0)).ravel() - mu ** 2
    FF = var / mu
    x, y = runningquantile(np.log(mu), np.log(FF / mu), fit_percentile, nBins)
    x, y = x[~np.isnan(y)], y[~np.isnan(y)]
    gLog = lambda inp: np.log(inp[1] * np.exp(-inp[0]) + inp[2])  # noqa: E731
    h, b = np.histogram(np.log(FF[mu > 0]), bins=200)
    b = b[:-1] + np.diff(b) / 2
    c = np.max((np.exp(b[np.argmax(h)]), 1))
    errFun = lambda b2: np.sum(abs(gLog([x, c, b2]) - y) ** error_wt)  # noqa: E731
    bb = scipy.optimize.fmin(func=errFun, x0=[0.1], disp=False)
    a = c / (1 + bb) - 1
    return FF / ((1 + a) * (1 + bb) + bb * mu), gix


V, gix = get_vscores(E)
ok = V > 0; V, gix = V[ok], gix[ok]
cnt = np.asarray((E[:, gix] >= 3).sum(0)).ravel()
hv = gix[(cnt >= 3) & (V >= np.percentile(V, 85))]
say("highly variable genes: %d" % len(hv))
seeds = [g for g in ["Ube2c", "Hmgb2", "Hmgn2", "Tuba1b", "Ccnb1", "Tubb5", "Top2a", "Tubb4b"] if g in genes]
sidx = np.array([genes.index(g) for g in seeds])
S = E[:, sidx].toarray(); S = S[:, S.sum(0) > 0]
cc = ((S - S.mean(0)) / S.std(0)).sum(1)
cc = (cc - cc.mean()) / cc.std()
H = E[:, hv].toarray()
Hz = (H - H.mean(0)) / np.where(H.std(0) > 0, H.std(0), 1)
corr = (Hz * cc[:, None]).mean(0)
keep = hv[corr < 0.1]
say("cell-cycle filter (seeds %s): %d of %d kept" % (seeds, len(keep), len(hv)))
del H, Hz

X = E[:, keep].toarray()
xs = sklearn.preprocessing.StandardScaler().fit_transform(X).astype(np.float32)
pca = sklearn.decomposition.PCA(n_components=50, random_state=0)
xp = pca.fit_transform(xs).astype(np.float32)
say("PCA variance kept %.3f" % float(pca.explained_variance_ratio_.sum()))
days = meta["Time point"].astype(float).values
y = sorted(set(days.tolist())); assert y == [2.0, 4.0, 6.0], y
state = meta["Cell type annotation"].astype(str).values

# KEGG growth (genes among the highly variable genes before the cell-cycle filter)
from annoy import AnnoyIndex  # noqa: E402
Xh = E[:, hv].toarray()
xh = sklearn.preprocessing.StandardScaler().fit_transform(Xh)
hv_names = [genes[i] for i in hv]
b = [g for g in pd.read_csv(os.path.join(KD, "mm_birth_kegg_mmu04110.csv"), index_col=0)["gene_symbol"] if g in hv_names]
d = [g for g in pd.read_csv(os.path.join(KD, "mm_death_kegg_mmu04210.csv"), index_col=0)["gene_symbol"] if g in hv_names]
b, d = [g for g in b if g not in d], [g for g in d if g not in b]
say("KEGG genes among the variable genes: birth %d %s, death %d %s" % (len(b), b, len(d), d))
col = {g: j for j, g in enumerate(hv_names)}
ay = AnnoyIndex(50, "euclidean")
for i in range(xp.shape[0]):
    ay.add_item(i, xp[i])
ay.build(10)
nbr = np.array([ay.get_nns_by_item(i, 20) for i in range(xp.shape[0])])


def smooth(raw, beta=0.1):
    prev = raw.copy(); cur = np.zeros(prev.shape)
    for _ in range(5):            # as theirs: prev and cur are one array after the first sweep
        for i in range(len(prev)):
            xn = prev[nbr[i]]
            cur[i] = beta * xn[0] + (1 - beta) * xn[1:].mean(axis=0)
        prev = cur
    return cur


bs = xh[:, [col[g] for g in b]].mean(1); ds = xh[:, [col[g] for g in d]].mean(1)
L0, L, k = 0.3, 1.1, 0.001
kb, kd = np.log(k) / np.min(bs), np.log(k) / np.min(ds)
g = (L0 + L / (1 + np.exp(-kb * smooth(bs)))) - (L0 + L / (1 + np.exp(-kd * smooth(ds))))
say("KEGG growth on day-2 cells: mean %.3f sd %.3f range %.3f to %.3f"
    % (g[days == 2].mean(), g[days == 2].std(), g[days == 2].min(), g[days == 2].max()))
for name, w0 in (("none", np.zeros(int((days == 2).sum()), dtype=np.float32)), ("kegg", g[days == 2].astype(np.float32))):
    torch.save(dict(xp=[torch.from_numpy(xp[days == dd]) for dd in y], y=y,
                    w=[w0] + [np.zeros(int((days == dd).sum()), dtype=np.float32) for dd in y[1:]],
                    celltype=state, tps=days, spring_x=meta["SPRING-x"].values, spring_y=meta["SPRING-y"].values,
                    genes=np.array([genes[i] for i in keep])), os.path.join(OUT, "data_full_%s.pt" % name))
np.save(os.path.join(OUT, "growth_kegg_all.npy"), g)
say("wrote data_full_none.pt and data_full_kegg.pt")
