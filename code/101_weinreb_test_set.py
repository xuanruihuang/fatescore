"""The published LARRY fate benchmark on its own cells: the 335 day-2 test cells of Weinreb et
al. 2020 (Fig. 5), also used by PRESCIENT (Yeo et al. 2021) and Chreode, scored with every
method here and with the connectivity kernel.

Test set, exactly as constructed in scDiffEq/LARRY-dataset (larry/tasks/fate_prediction/
_weinreb_test_set.py, "335 cell test set used in Weinreb, 2020 and Yeo, 2021") from the files
Weinreb et al. shared for Fig. 5: day-2 cells on the neutrophil/monocyte trajectory, early,
not in the held-out half, with at least one neutrophil or monocyte sister. Target: the
neutrophil share among the cell's neutrophil and monocyte sisters. Metrics as published:
Pearson r with that share, and AUROC for share > 0.5 (PRESCIENT).

The files index cells in the order of the full in vitro data (130,887 cells). With the GEO
metadata of that data (GSM4185642, same row order), cells are matched to this directory's object
by library and cell barcode, which is exact. Without it, they are matched by SPRING coordinates
rounded to two decimals; the Fig. 5 coordinate files differ from the metadata by up to 5e-4, so
that route misses cells whose coordinates round differently (298 of 335 matched).

Usage: 101_weinreb_test_set.py <fig5_dir> <h5ad> <featdir> <d2_labels.tsv> <out.csv> [<metadata.txt.gz>]
"""
import glob
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

F5, H5, FD, LAB, OUT = sys.argv[1:6]
META = sys.argv[6] if len(sys.argv) > 6 else None
ld = lambda n: np.load(os.path.join(F5, n + ".npy"), allow_pickle=True)  # noqa: E731
tp, nm, early, held = ld("timepoints"), ld("neutrophil_monocyte_trajectory_mask"), ld("early_cells"), ld("heldout_mask")
cfm = ld("clonal_fate_matrix")
sx, sy = ld("coordinates_x"), ld("coordinates_y")
preds = {k: ld(k + "_predictions") for k in ["PBA", "FateID", "WOT"]}
smooth = ld("smoothed_groundtruth_from_heldout")
names = open(os.path.join(F5, "FINAL_fate_names.txt")).read().split()
print("shapes: tp %s nm %s early %s held %s cfm %s preds %s smooth %s; fates %s"
      % (tp.shape, nm.shape, early.shape, held.shape, cfm.shape, preds["WOT"].shape, smooth.shape, names))

# --- the construction of LARRY-dataset's WeinrebTestSet, step by step
nm2 = nm[tp == 2]                       # trajectory mask over day-2 cells
e_nm = early[nm2]                       # early flag of the trajectory day-2 cells
keep = ~held                            # held-out flag over those cells
nm_counts = cfm[:, 5:7].sum(1)
has_fate = (nm_counts > 0) & nm2
has_fate_eh = has_fate[nm2 & early][keep[e_nm]]
d2_orig = np.where(tp == 2)[0]
test_orig = d2_orig[nm2 & early][keep[e_nm]][has_fate_eh]
nm_pct = cfm[has_fate, 5] / cfm[has_fate, 5:7].sum(1)
y = nm_pct[(e_nm & keep)[has_fate[nm2]]]
sub = lambda p: p[e_nm][keep[e_nm]][has_fate_eh]  # noqa: E731
pub = {k: sub(v) for k, v in preds.items()}
pub["held-out clonal (upper bound)"] = smooth[e_nm & keep][has_fate_eh]
print("test cells %d (published 335); fate columns 5,6 = %s, %s" % (len(test_orig), names[5], names[6]))
assert len(y) == len(test_orig)

# --- match to this directory's object (by barcode with the metadata, else by SPRING coordinates)
obs = ad.read_h5ad(H5, backed="r").obs
key = lambda x, y_: np.round(x, 2).astype(str).astype(object) + "_" + np.round(y_, 2).astype(str).astype(object)  # noqa: E731
okey = pd.Series(obs.index.values, index=key(obs["SPRING-x"].values.astype(float), obs["SPRING-y"].values.astype(float)))
okey = okey[~okey.index.duplicated(keep=False)]
tkey = key(sx[test_orig].astype(float), sy[test_orig].astype(float))
cell = np.array([okey.get(k, None) for k in tkey], dtype=object)
if META is not None:
    meta = pd.read_csv(META, sep="\t")
    assert len(meta) == len(sx)
    bc = (meta["Library"].astype(str) + ":" + meta["Cell barcode"].astype(str).str.replace("-", "", regex=False)).values
    oset = set(obs.index)
    cell = np.array([b if b in oset else None for b in bc[test_orig]], dtype=object)
    print("matched by barcode (metadata); by rounded coordinates it would be %d"
          % sum(okey.get(k, None) is not None for k in tkey))
T = pd.read_csv(LAB, sep="\t").set_index("cell")
d2_names = json.load(open(os.path.join(FD, "d2_names.json")))
d2_names = d2_names if isinstance(d2_names, list) else d2_names.get("names", d2_names)
pos = {n: i for i, n in enumerate(d2_names)}
in_obj = np.array([c is not None for c in cell])
in_d2 = np.array([c is not None and c in pos for c in cell])
print("matched to the object %d / %d; among this directory's early cells %d" % (in_obj.sum(), len(cell), in_d2.sum()))
# consistency: our own clonal share for the matched cells against the published target
ours = T.loc[[c for c in cell[in_d2]], "f_neu"].values.astype(float)
print("our neutrophil share vs the published target on matched cells: r = %.4f, max |diff| = %.3f"
      % (pearsonr(ours, y[in_d2])[0], np.nanmax(np.abs(ours - y[in_d2]))))

idx = np.array([pos[c] for c in cell[in_d2]])
yt = y[in_d2]
clones = T.loc[[c for c in cell[in_d2]], "clone"].values
ucl, inv = np.unique(clones, return_inverse=True)
print("matched test cells %d in %d clones" % (len(yt), len(ucl)))


def share(name):
    M = np.load(os.path.join(FD, name + ".npy"))
    if M.ndim == 2 and M.shape[1] > 1:
        s = M[:, 0] + M[:, 1]
        return np.where(s > 0, M[:, 0] / np.where(s > 0, s, 1), 0.5)
    return M.ravel()


scores = {}
for k, v in pub.items():
    scores["published: " + k] = v[in_d2]
for f in sorted(glob.glob(os.path.join(FD, "nm_*.npy"))):
    scores[os.path.basename(f)[3:-4]] = share(os.path.basename(f)[:-4])[idx]
REF = "cr_conn_n30_p30_d1"
rng = np.random.default_rng(97)
# bootstrap over clones: resample clones, take all their test cells
members = [np.where(inv == j)[0] for j in range(len(ucl))]
boots = [np.concatenate([members[j] for j in rng.integers(0, len(ucl), len(ucl))]) for _ in range(2000)]
lab = (yt > 0.5).astype(int)


def stats(s):
    return pearsonr(s, yt)[0], roc_auc_score(lab, s)


rows = []
ref = scores[REF]
for k, s in scores.items():
    if np.std(s) == 0:
        continue
    r, a = stats(s)
    dr, da = [], []
    for b in boots:
        if np.std(s[b]) == 0 or lab[b].min() == lab[b].max():
            continue
        dr.append(pearsonr(s[b], yt[b])[0] - pearsonr(ref[b], yt[b])[0])
        da.append(roc_auc_score(lab[b], s[b]) - roc_auc_score(lab[b], ref[b]))
    rows.append(dict(method=k, r=r, auroc=a, r_minus_kernel=r - stats(ref)[0],
                     r_lo=np.percentile(dr, 2.5), r_hi=np.percentile(dr, 97.5),
                     auroc_minus_kernel=a - stats(ref)[1],
                     auroc_lo=np.percentile(da, 2.5), auroc_hi=np.percentile(da, 97.5),
                     n_cells=len(yt), n_clones=len(ucl)))
R = pd.DataFrame(rows).sort_values("r", ascending=False)
R.to_csv(OUT, index=False)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
# the published methods on all 335 cells, for comparison with the papers' numbers
for k, v in pub.items():
    print("all %d cells, published %s: r = %.3f, AUROC = %.3f" % (len(y), k, pearsonr(v, y)[0], roc_auc_score((y > 0.5).astype(int), v)))
