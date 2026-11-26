"""Positive control for an arm's expression block: can it predict labels it
obviously should? If an arm's clonal fate comes out at chance, the first thing
to rule out is that the block or the row alignment is broken."""
import json, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

featdir, labels = sys.argv[1], sys.argv[2]
T = pd.read_csv(labels, sep="\t")
d2 = json.load(open(featdir + "/d2_names.json"))
assert list(T["cell"]) == d2
X = np.load(featdir + "/expr_hvg.npy")
print("block %s, labels %s" % (X.shape, T.shape))
print("nonzero fraction %.4f, row sums: min %.3f median %.3f max %.3f"
      % ((X > 0).mean(), X.sum(1).min(), np.median(X.sum(1)), X.sum(1).max()))

T["library"] = [c.split(":")[0] for c in T["cell"]]
tasks = {}
libs = T["library"].value_counts()
if len(libs) >= 2:
    top = libs.index[0]
    tasks["library %s vs rest" % top] = (T["library"] == top).astype(int).values
if T["pop"].nunique() >= 2:
    top = T["pop"].value_counts().index[0]
    tasks["starting population %s vs rest" % top] = (T["pop"] == top).astype(int).values
sel = (T["n_sis"] >= 3) & T["dom"].notna() & (T["dom"] != "")
groups = T["clone"].values
for name, y in tasks.items():
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 20:
        print("  %-42s skipped (too few of one class)" % name); continue
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        p = PCA(n_components=50, random_state=0).fit(X[tr])
        s = StandardScaler().fit(p.transform(X[tr]))
        m = LogisticRegression(max_iter=2000, C=1.0).fit(s.transform(p.transform(X[tr])), y[tr])
        oof[te] = m.predict_proba(s.transform(p.transform(X[te])))[:, 1]
    print("  %-42s AUROC %.3f  (n=%d, positives %d)"
          % (name, roc_auc_score(y, oof), len(y), int(y.sum())))
