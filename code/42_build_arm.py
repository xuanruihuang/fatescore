"""Build the in vivo and cytokine arms of the same experiment into our format.

The manuscript's whole evaluation runs on one arm of Weinreb et al.'s state-fate
experiment, the in vitro one, because that is the arm whose spliced and unspliced
counts somebody has made public. The other two arms -- cells transplanted into
mice, and cells grown in five cytokine conditions -- are deposited as normalised
counts only, so no velocity method can be run on them at all. Everything that
does not need a splicing layer can: the expression kernels, the
distance-to-centroid baseline, and the snapshot transport methods.

That is the point of building them. A benchmark that lives on one arm cannot
tell a property of the method from a property of the arm, and the cytokine arm
happens to carry a neutrophil-versus-monocyte task of almost the same size as
the in vitro one, while the in vivo arm's clones are dominated by erythroid
fate instead. So the three arms give one near-replicate and one genuinely
different task.

What this writes, per arm, is exactly the interface the existing scripts read:
an h5ad with time_info, state_info and X_clone; a feature directory with
hvg.json, d2_names.json and cellrank_log.json; and a labels table with the
columns 30_kernel_eval.py and cvlib use (cell, clone, pop, n_sis, dom, plus the
batch split). No splicing layer and no velocity, by necessity.

The tie-breaking order for a clone's dominant fate is fixed here, before any
label is built, as the order of the arm's late-cell counts in
out/larry_<arm>_feas.csv, which 26_larry_arms_feasibility.py computed from the
raw files. For the in vitro arm the same rule's sensitivity was measured in
24_tie_rule_sensitivity.py.

Usage:
  python 42_build_arm.py --arm invivo  --data <dir> --meta <dir> --out <dir>
  python 42_build_arm.py --arm cytokine --data <dir> --meta <dir> --out <dir>
"""
import argparse
import gzip
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

ARMS = {
    "invivo": dict(
        gsm="GSM4185643_stateFate_inVivo",
        time_col="Time point",
        state_col="Cell type annotation",
        pop_col="Starting population",
        batch_col="Recipient mouse",
        early=2,
        # fixed in advance: descending late-cell count in the feasibility table
        mature=["Ery", "B", "DC", "Neu", "Mono", "Baso", "T", "NK"],
    ),
    "cytokine": dict(
        gsm="GSM4185644_stateFate_cytokinePerturbation",
        time_col="Time point",
        state_col="Cell type annotation",
        pop_col=None,          # one starting population in this arm
        batch_col="Cytokine condition",
        early=2,
        mature=["Mo", "Neu", "Ba", "Er", "Mk", "cDC", "Ma", "pDC1", "Eos",
                "pDC2"],
    ),
}


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def lines(path):
    """Read a gzipped text file's lines. These deposits end without a final
    newline, so the last cell or gene is lost by anything that counts newlines;
    splitting the decoded text keeps it."""
    with gzip.open(path, "rt") as fh:
        txt = fh.read()
    out = txt.split("\n")
    if out and out[-1] == "":
        out.pop()
    return out


def read_mtx(path, n_cells, n_genes):
    say("reading %s" % os.path.basename(path))
    with gzip.open(path, "rb") as fh:
        M = sio.mmread(fh)
    M = sp.csr_matrix(M, dtype=np.float32)
    if M.shape == (n_genes, n_cells):
        M = M.T.tocsr()
    assert M.shape == (n_cells, n_genes), (M.shape, n_cells, n_genes)
    say("matrix %s, %d nonzeros" % (M.shape, M.nnz))
    return M


def read_clones(path, n_cells):
    with gzip.open(path, "rb") as fh:
        C = sp.csr_matrix(sio.mmread(fh))
    if C.shape[0] == n_cells:
        pass
    elif C.shape[1] == n_cells:
        C = C.T.tocsr()
    else:
        raise AssertionError("clone matrix %s does not fit %d cells"
                             % (C.shape, n_cells))
    say("clone matrix %s (cells x clones)" % (C.shape,))
    return C


def dominant(counts, mature):
    """The clone's most common mature annotation; ties go to the earlier entry
    of the fixed order. Same rule as 01_counts.py and 02b_labels.py."""
    best, bestn = None, -1
    for st in mature:
        n = int(counts.get(st, 0))
        if n > bestn:
            best, bestn = st, n
    return best if bestn > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--data", required=True, help="dir with the counts mtx")
    ap.add_argument("--meta", required=True, help="dir with metadata and clones")
    ap.add_argument("--out", required=True, help="dir for the h5ad and outputs")
    ap.add_argument("--featdir", default="")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--downsample-late", type=int, default=0,
                    help="cells to keep per timepoint after the first; "
                         "0 keeps everything. Every early cell is always "
                         "kept, so the labels and the evaluation set do "
                         "not change")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the downsampling")
    ap.add_argument("--suffix", default="",
                    help="appended to the feature directory and h5ad "
                         "names, e.g. _ds21000")
    ap.add_argument("--min-cells", type=int, default=30,
                    help="cells an annotation needs to be a terminal state")
    a = ap.parse_args()
    cfg = ARMS[a.arm]
    gsm = cfg["gsm"]
    featdir = a.featdir or os.path.join(a.out, "feat_%s%s"
                                    % (a.arm, a.suffix))
    os.makedirs(featdir, exist_ok=True)

    genes = lines(os.path.join(a.data, gsm + "_gene_names.txt.gz"))
    meta = pd.read_csv(os.path.join(a.meta, gsm + "_metadata.txt.gz"),
                       sep="\t", compression="gzip")
    say("%d genes, metadata %s" % (len(genes), meta.shape))
    n_cells = meta.shape[0]

    M = read_mtx(os.path.join(a.data, gsm + "_normed_counts.mtx.gz"),
                 n_cells, len(genes))
    C = read_clones(os.path.join(a.meta, gsm + "_clone_matrix.mtx.gz"), n_cells)

    names = (meta["Library"].astype(str) + ":"
             + meta["Cell barcode"].astype(str)).values
    A = ad.AnnData(M, obs=pd.DataFrame(index=pd.Index(names, name="cell")),
                   var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    # 33 library-and-barcode pairs occur twice in the in vivo deposit (67
    # cells). None of them is an early cell, so nothing downstream was
    # wrong, but a duplicated name makes obs_names.get_loc return a mask
    # instead of a position, which would corrupt a feature block without
    # any error. Make them unique here, once, before anything reads them.
    if not A.obs_names.is_unique:
        dup = int(A.obs_names.duplicated(keep=False).sum())
        A.obs_names_make_unique()
        say("made %d non-unique cell names unique" % dup)
    A.obs["time_info"] = pd.to_numeric(meta[cfg["time_col"]].values)
    A.obs["state_info"] = meta[cfg["state_col"]].astype(str).values
    A.obs["pop"] = (meta[cfg["pop_col"]].astype(str).values
                    if cfg["pop_col"] else "all")
    A.obs["batch"] = meta[cfg["batch_col"]].astype(str).values
    A.obsm["X_clone"] = C
    # the deposit gives one clone per cell at most; -1 means none
    has = np.asarray((C > 0).sum(1)).ravel()
    clone = np.asarray(C.argmax(1)).ravel()
    clone = np.where(has > 0, clone, -1)
    A.obs["clone"] = clone
    say("cells with a clone: %d; clones present: %d"
        % (int((clone >= 0).sum()), len(set(clone[clone >= 0].tolist()))))
    say("annotations: %s" % A.obs["state_info"].value_counts().to_dict())

    # --- labels ---------------------------------------------------------
    mature = cfg["mature"]
    obs = A.obs.copy()
    obs["cell"] = A.obs_names
    obs["mature"] = obs["state_info"].isin(mature)
    early_t = cfg["early"]
    late = obs[(obs["time_info"] > early_t) & obs["mature"]
               & (obs["clone"] >= 0)]
    comp = (late.groupby(["clone", "state_info"], observed=True).size()
            .unstack(fill_value=0))
    dom = {int(cl): dominant(row, mature) for cl, row in comp.iterrows()}
    nsis = late.groupby("clone").size().to_dict()

    early = obs[(obs["time_info"] == early_t) & (obs["clone"] >= 0)].copy()
    early["n_sis"] = [int(nsis.get(int(c), 0)) for c in early["clone"]]
    early["dom"] = [dom.get(int(c)) or "" for c in early["clone"]]
    # per-batch split, the analogue of the in vitro arm's two wells: the number
    # of mature siblings and the dominant fate within each of the two batches
    # that carry the most of the clone's siblings.
    bsis, bdom = {}, {}
    for (cl, b), g in late.groupby(["clone", "batch"], observed=True):
        bsis.setdefault(int(cl), []).append((len(g), str(b)))
    for cl, items in bsis.items():
        items.sort(reverse=True)
        bdom[cl] = items
    for j in (1, 2):
        early["n_sis_b%d" % j] = [
            bdom.get(int(c), [(0, "")] * 2)[j - 1][0]
            if len(bdom.get(int(c), [])) >= j else 0 for c in early["clone"]]
        early["batch_b%d" % j] = [
            bdom.get(int(c), [(0, "")] * 2)[j - 1][1]
            if len(bdom.get(int(c), [])) >= j else "" for c in early["clone"]]
    cb = {}
    for (cl, b, st), n in (late.groupby(["clone", "batch", "state_info"],
                                        observed=True).size().items()):
        cb.setdefault((int(cl), str(b)), {})[str(st)] = int(n)
    cb_dom = {k: (dominant(v, mature) or "") for k, v in cb.items()}
    for j in (1, 2):
        col = []
        for c in early["clone"]:
            items = bdom.get(int(c), [])
            col.append("" if len(items) < j
                       else cb_dom.get((int(c), items[j - 1][1]), ""))
        early["dom_b%d" % j] = col

    # how many mature annotations tie for the clone's top count: a tie
    # means the dominant-fate label was decided by the fixed order, not
    # by the data, and the in vivo arm has many such clones
    tie = {}
    for cl, row in comp.iterrows():
        v = row.reindex(mature).fillna(0).values
        tie[int(cl)] = int((v == v.max()).sum()) if v.max() > 0 else 0
    early["n_top_tie"] = [int(tie.get(int(c), 0)) for c in early["clone"]]
    T = early[["cell", "clone", "pop", "state_info", "n_sis", "dom",
               "n_top_tie",
               "n_sis_b1", "dom_b1", "batch_b1",
               "n_sis_b2", "dom_b2", "batch_b2"]].copy()
    T = T.rename(columns={"state_info": "state_d2"})
    lab_path = os.path.join(a.out, "%s_labels.tsv" % a.arm)
    T.to_csv(lab_path, sep="\t", index=False)
    say("labels %s -> %s" % (T.shape, lab_path))
    sel = T[(T["n_sis"] >= 3) & (T["dom"] != "")]
    say("early cells with >=3 mature siblings: %d, in %d clones"
        % (len(sel), sel["clone"].nunique()))
    say("clone-level dominant fate: %s"
        % sel.groupby("clone")["dom"].first().value_counts().to_dict())

    # --- optional downsampling of the later timepoints ------------------
    # Two of the four snapshot methods materialise a dense coupling, so at the
    # in vivo arm's full resolution their peak memory runs into hundreds of
    # gigabytes. Downsampling cells per timepoint is the ordinary way round
    # that and is what Waddington-OT's own paper does. Every early cell is
    # kept, so the evaluation set and the labels are untouched: the labels come
    # from the full experiment above, and only what the methods see shrinks.
    if a.downsample_late > 0:
        rng = np.random.default_rng(a.seed)
        keep = np.asarray(A.obs["time_info"].values == early_t)
        for t in sorted(set(A.obs["time_info"].values.tolist())):
            if t == early_t:
                continue
            idx = np.where(np.asarray(A.obs["time_info"].values == t))[0]
            take = (idx if idx.size <= a.downsample_late
                    else rng.choice(idx, a.downsample_late, replace=False))
            keep[take] = True
            say("timepoint %g: %d cells -> %d kept" % (t, idx.size, take.size))
        A = A[keep].copy()
        say("downsampled object %s (seed %d); every early cell kept"
            % (A.shape, a.seed))
        assert int((A.obs["time_info"].values == early_t).sum()) == int(
            (obs["time_info"].values == early_t).sum())

    # --- features -------------------------------------------------------
    # These deposits are already library-size normalised, so only the log and
    # the gene selection are applied here; there is no splicing layer to
    # filter on, which is the one place this differs from 02_features.py.
    L = A.copy()
    sc.pp.log1p(L)
    sc.pp.highly_variable_genes(L, n_top_genes=a.n_hvg, flavor="seurat")
    hvg = L.var_names[L.var["highly_variable"].values].tolist()
    json.dump(hvg, open(os.path.join(featdir, "hvg.json"), "w"))
    d2_names = T["cell"].tolist()
    json.dump(d2_names, open(os.path.join(featdir, "d2_names.json"), "w"))

    # The expression baseline, as a feature block over the early cells:
    # the arm has no velocity, so this is what every other arm on it is
    # compared against, the way the in vitro arm compares against its
    # velocity kernel. Log-normalised counts over the same genes.
    d2_idx = np.array([A.obs_names.get_loc(n) for n in d2_names])
    Xb = L[d2_idx][:, hvg].X
    Xb = np.asarray(Xb.todense() if sp.issparse(Xb) else Xb,
                    dtype=np.float32)
    np.save(os.path.join(featdir, "expr_hvg.npy"), Xb)
    say("expression baseline block %s -> expr_hvg.npy" % (Xb.shape,))
    del L

    state = A.obs["state_info"].values
    present = [s for s in mature if (state == s).sum() >= a.min_cells]
    terminal = {s: int((state == s).sum()) for s in present}
    n_term = int(np.isin(state, present).sum())
    json.dump(dict(terminal_states=terminal, n_terminal=n_term,
                   n_transient=int(A.n_obs - n_term),
                   n_cells_in_object=int(A.n_obs),
                   note="built by 42_build_arm.py; no velocity on this arm"),
              open(os.path.join(featdir, "cellrank_log.json"), "w"), indent=1)
    say("terminal states %s" % terminal)
    # 38_scope_check.py reads this one: the number of cells every array
    # in this directory was built on, so a mismatch is an assertion
    # failure rather than something to notice later.
    json.dump(dict(n_cells_in_object=int(A.n_obs), arm=a.arm,
                   n_early=len(d2_names), n_hvg=len(hvg),
                   downsample_late=a.downsample_late, seed=a.seed),
              open(os.path.join(featdir, "scope.json"), "w"), indent=1)
    # log.json is what 38_scope_check.py keys a directory on, so write
    # the same fields 02_features.py writes for the in vitro arm.
    json.dump(dict(n_cells_total=int(A.n_obs), n_d2=len(d2_names),
                   n_hvg=len(hvg), arm=a.arm,
                   note="no splicing layer in this arm"),
              open(os.path.join(featdir, "log.json"), "w"), indent=1)

    h5 = os.path.join(a.out, "larry_%s%s.h5ad" % (a.arm, a.suffix))
    A.write_h5ad(h5, compression="gzip")
    say("wrote %s (%d cells, %d genes, %d highly variable)"
        % (h5, A.n_obs, A.n_vars, len(hvg)))
    json.dump(dict(arm=a.arm, suffix=a.suffix,
                   downsample_late=a.downsample_late, seed=a.seed,
                   n_cells=int(A.n_obs), n_genes=int(A.n_vars),
                   n_hvg=len(hvg), early_time=early_t,
                   n_early_with_clone=len(T), mature_order=mature,
                   tie_rule="fixed order above, declared before labelling",
                   labels=lab_path, h5ad=h5, featdir=featdir),
              open(os.path.join(a.out, "%s_build_log.json" % a.arm), "w"),
              indent=1)
    say("done")


if __name__ == "__main__":
    main()
