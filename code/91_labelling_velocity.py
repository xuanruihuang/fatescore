"""Agreement with velocity from metabolic labelling, against baselines without dynamics.

Velocity estimated from metabolically labelled (new) RNA is used as a reference for
splicing velocity: in dynamo (Qiu et al. 2022) the labelling velocity of the
haematopoietic scNT-seq data is the field the splicing velocity is compared with.
Here both are computed on the same cells and genes, and the splicing velocity is
compared with fields that contain no information about how cells change:

  degradation only     -gamma * Ms (gamma from the splicing steady-state fit)
  shuffled unspliced   Mu[perm] - gamma * Ms
  neighbour average    mean Ms of a cell's neighbours minus its own Ms

Labelling velocity: dynamo's one-shot pipeline (Preprocessor 'monocle' recipe,
tkey = labelling time, experiment_type 'one-shot'; tl.dynamics), layer velocity_T.
Splicing velocity: scVelo steady-state ('deterministic'), stochastic and
dynamical, on the spliced/unspliced layers of the same cells.
All fields are restricted to the genes they share and scored per cell by the
cosine with the labelling velocity (in gene space) and, after projection into the
data's UMAP with scVelo's velocity_graph / velocity_embedding on the same graph,
by the cosine of the projected 2-D vectors. Means over cells with cell-bootstrap
intervals (2,000), paired differences against the best splicing velocity.

Also reported: the cosine of the labelling velocity with its own degradation term
(-gamma_labelling * total), i.e. how much of the reference is degradation.

Usage: 91_labelling_velocity.py <h5ad> <out_prefix> [--tkey time --exp one-shot]
"""
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("out")
ap.add_argument("--tkey", default="time")
ap.add_argument("--exp", default="one-shot")
ap.add_argument("--n-top", type=int, default=2000)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--time-scale", type=float, default=1.0)
ap.add_argument("--cycle-key", default=None)
ap.add_argument("--cycle-window", type=float, default=0.15)
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--recipe", default="generic", choices=["generic", "hsc_tutorial", "published", "kin_tutorial", "organoid_tutorial"])
ap.add_argument("--embed-cols", default=None, help="obs columns used as the 2-D embedding instead of UMAP")
ap.add_argument("--all-genes", action="store_true", help="do not require scVelo's steady-state fit flag")
ap.add_argument("--gene-map", default=None, help="TSV gene_id -> symbol (91b_gene_symbols.py)")
ap.add_argument("--ref", default=None, help="published labelling velocity (cells x genes, var gene_id, time_<t>_gamma)")
a = ap.parse_args()

import anndata as ad  # noqa: E402
import dynamo as dyn  # noqa: E402
import scvelo as scv  # noqa: E402

R = ad.read_h5ad(a.h5ad)
R.obs[a.tkey] = R.obs[a.tkey].astype(float) / a.time_scale
for k in list(R.layers.keys()):
    if k is None or str(k) == "None":
        del R.layers[k]
if a.gene_map:
    gm = pd.read_csv(a.gene_map, sep="\t", dtype=str).fillna("")
    gm = dict(zip(gm["gene_id"], gm["symbol"]))
    R.var["gene_name_mapping"] = [gm.get(i.split(".")[0], "") for i in R.var_names]
if "gene_name_mapping" in R.var.columns:
    # what dynamo's gene-name conversion does with this column: Ensembl IDs -> symbols
    sym = R.var["gene_name_mapping"].astype(str).values
    R.var["gene_id"] = R.var_names.values
    R.var_names = np.where(~np.isin(sym, ["nan", "", "None"]), sym, R.var_names.values)
    R.var_names_make_unique()
if "total" not in R.layers:
    R.layers["total"] = R.X.copy()
# counts stored as integers would make dynamo's in-place normalisation fail
R.X = R.X.astype(np.float32)
for k in list(R.layers.keys()):
    R.layers[k] = R.layers[k].astype(np.float32)
say("data %s, layers %s" % (R.shape, list(R.layers.keys())))

# ---- labelling velocity (dynamo) ----
def dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M)


L = R.copy()
GAMT = None
if a.recipe == "published":
    # the labelling velocity published with dynamo for these data (total RNA velocity
    # alpha - gamma * s, layer velocity_alpha_minus_gamma_s of dynamo's processed object)
    P = ad.read_h5ad(a.ref)
    cells = [c for c in P.obs_names if c in set(R.obs_names)]
    R = R[cells].copy()
    P = P[cells].copy()
    pos = pd.Index(R.var_names).get_indexer(P.var["gene_name"].astype(str).values)
    okg = pos >= 0
    dg = R.var_names[pos[okg]]
    VT = np.asarray(P.X, dtype=float)[:, okg]
    tt = P.obs[a.tkey].astype(float).values
    GAMT = np.zeros_like(VT)
    for t0 in np.unique(tt):
        GAMT[tt == t0] = np.nan_to_num(P.var["time_%d_gamma" % int(t0)].values[okg].astype(float))
    R.obsm["X_umap"] = np.asarray(P.obsm["X_umap"])
    DEG = None
    say("published velocity: %d cells, %d of %d genes matched" % (len(cells), okg.sum(), P.n_vars))
elif a.recipe == "hsc_tutorial":
    # exactly the calls of dynamo's tutorial for these data (301_tutorial_hsc_velocity):
    # total RNA velocity alpha - gamma * s, alpha from the new RNA of each labelling
    # duration, gamma fitted per labelling duration, s the smoothed spliced RNA
    use = L.uns["genes_to_use"]
    pre = dyn.pp.Preprocessor(force_gene_list=use)
    pre.convert_gene_name = lambda adata, *args, **kw: adata  # gene IDs kept as given
    pre.config_monocle_recipe(L, n_top_genes=len(use))
    pre.preprocess_adata_monocle(L, tkey=a.tkey, experiment_type=a.exp)
    dyn.tl.reduceDimension(L)
    dyn.tl.moments(L, group=a.tkey)
    L.uns["pp"]["has_splicing"] = False
    dyn.tl.dynamics(L, group=a.tkey, one_shot_method="sci_fate", model="deterministic")
    dyn.tl.calculate_velocity_alpha_minus_gamma_s(L)
    gmask = L.var["use_for_pca"].values.astype(bool)
    dg = L.var_names[gmask]
    VT = dense(L.layers["velocity_alpha_minus_gamma_s"])[:, gmask]
    MS = dense(L.layers["M_s"])[:, gmask]
    from dynamo.tools.utils import get_vel_params
    DEG = np.zeros_like(MS)
    tt = L.obs[a.tkey].astype(float).values
    for t0 in np.unique(tt):
        g = np.asarray(get_vel_params(L, params="gamma", genes=L.var["use_for_pca"],
                                      kin_param_pre="time_%d_" % int(t0)), dtype=float).ravel()
        DEG[tt == t0] = -np.nan_to_num(g) * MS[tt == t0]
        say("labelling time %g: %d cells, gamma median %.3g" % (t0, (tt == t0).sum(), np.nanmedian(g)))
else:
    for k in ("spliced", "unspliced"):
        if k in L.layers:
            del L.layers[k]
    if a.recipe == "kin_tutorial":
        # the call of dynamo's scEU-seq tutorial (302_scEU_seq_rpe1_analysis_kinetic)
        dyn.tl.recipe_kin_data(adata=L, keep_filtered_genes=True, keep_raw_layers=True,
                               del_2nd_moments=False, tkey=a.tkey)
    elif a.recipe == "organoid_tutorial":
        # the calls of dynamo's organoid tutorial (303_scEU_seq_organoid_analysis_kinetic):
        # pulse cells only, time in hours, monocle recipe on 1,000 genes, deterministic model
        dyn.pp.recipe_monocle(L, n_top_genes=1000, total_layers=False)
        dyn.tl.dynamics(L, model="deterministic", tkey=a.tkey, assumption_mRNA="ss")
    else:
        pre = dyn.pp.Preprocessor()
        pre.convert_gene_name = lambda adata, *args, **kw: adata  # gene IDs kept as given
        pre.preprocess_adata(L, recipe="monocle", tkey=a.tkey, experiment_type=a.exp)
        dyn.tl.dynamics(L)
    dg = L.var_names[L.var["use_for_dynamics"].values.astype(bool)]
    VT = dense(L[:, dg].layers["velocity_T"])
    MT = dense(L[:, dg].layers["M_t"])
    k = list(L.uns["vel_params_names"]).index("gamma")
    gam_lab = np.nan_to_num(np.asarray(L[:, dg].varm["vel_params"], dtype=float)[:, k])
    DEG = -gam_lab * MT
say("labelling velocity genes %d, nonzero cells %d" % (len(dg), int((np.abs(np.nan_to_num(VT)).sum(1) > 0).sum())))

# ---- splicing velocity (scVelo) ----
S = ad.AnnData(R.layers["spliced"].copy(), obs=R.obs.copy(), var=R.var.copy())
S.layers["spliced"] = R.layers["spliced"].copy()
S.layers["unspliced"] = R.layers["unspliced"].copy()
if a.embed_cols:
    # a measured embedding (e.g. FUCCI fluorescence); kept under the name scVelo projects into
    S.obsm["X_umap"] = R.obs[a.embed_cols.split(",")].values.astype(float)
elif "X_umap" in R.obsm:
    S.obsm["X_umap"] = R.obsm["X_umap"].copy()
import scanpy as sc  # noqa: E402
# normalisation as in 02_features.py, but on the genes dynamo selected for the
# labelling velocity, so that both velocities are estimated on the same genes
scv.pp.normalize_per_cell(S)
sc.pp.log1p(S)
su = np.asarray((S.layers["unspliced"] > 0).sum(0)).ravel()
ss = np.asarray((S.layers["spliced"] > 0).sum(0)).ravel()
S = S[:, S.var_names.isin(dg) & (su >= 10) & (ss >= 10)].copy()
say("dynamo genes with spliced and unspliced reads in >= 10 cells: %d of %d" % (S.n_vars, len(dg)))
scv.pp.moments(S, n_pcs=30, n_neighbors=30)
if "X_umap" not in S.obsm:
    sc.tl.umap(S, random_state=0)
scv.tl.velocity(S, mode="deterministic", vkey="v_ss")
scv.tl.velocity(S, mode="stochastic", vkey="v_st")
scv.tl.recover_dynamics(S, n_jobs=a.jobs)
scv.tl.velocity(S, mode="dynamical", vkey="v_dyn")

genes = [g for g in S.var_names if g in set(dg)]
gi = S.var_names.get_indexer(genes)
li = pd.Index(dg).get_indexer(genes)
VT = np.where(np.isfinite(VT), VT, np.nan)
ok = (np.isfinite(np.asarray(S.var["v_ss_gamma"], dtype=float)[gi]) &
      (np.asarray(S.var["v_ss_genes"]).astype(bool)[gi] | a.all_genes) &
      np.isfinite(VT[:, li]).all(0))
say("genes: steady-state fit %d, dynamical fit %d, finite labelling velocity %d" % (
    np.asarray(S.var["v_ss_genes"]).astype(bool)[gi].sum(),
    np.asarray(S.var["v_dyn_genes"]).astype(bool)[gi].sum(),
    np.isfinite(VT[:, li]).all(0).sum()))
genes = [g for g, o in zip(genes, ok) if o]
gi = S.var_names.get_indexer(genes)
li = pd.Index(dg).get_indexer(genes)
say("shared genes %d" % len(genes))

Ms = np.asarray(S.layers["Ms"])[:, gi]
Mu = np.asarray(S.layers["Mu"])[:, gi]
gam = np.asarray(S.var["v_ss_gamma"], dtype=float)[gi]
D = S.obsp["distances"].tocsr()
nbr = [D.indices[D.indptr[i]:D.indptr[i + 1]] for i in range(S.n_obs)]
perm = np.random.default_rng(91).permutation(S.n_obs)
knn = np.vstack([Ms[n].mean(0) - Ms[i] for i, n in enumerate(nbr)])
ref = VT[:, li]
fields = {
    "steady_state": np.nan_to_num(np.asarray(S.layers["v_ss"])[:, gi]),
    "stochastic": np.nan_to_num(np.asarray(S.layers["v_st"])[:, gi]),
    "dynamical": np.nan_to_num(np.asarray(S.layers["v_dyn"])[:, gi]),
    "degradation_only": -gam * Ms,
    "shuffled_unspliced": Mu[perm] - gam * Ms,
    "neighbour_average": knn,
}
if GAMT is not None:
    # degradation term of the reference itself, -gamma_t * s, with s the smoothed spliced
    # counts of this pipeline
    fields["labelling_degradation_term"] = -GAMT[:, li] * Ms
elif DEG is not None:
    fields["labelling_degradation_term"] = DEG[:, li]


def rowcos(A, B):
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    c = (A * B).sum(1) / np.where(na * nb > 0, na * nb, np.nan)
    return c


refs = {"labelling": ref}
if "labelling_degradation_term" in fields:
    # the production part of the reference: labelling velocity minus its own degradation
    # term, i.e. the part that comes from newly made RNA, so that agreement through a
    # shared -gamma s term is not counted
    refs["production"] = ref - fields["labelling_degradation_term"]
cyc_emb = None
if a.cycle_key:
    # measured direction, independent of the transcriptome: in the measured embedding
    # (FUCCI fluorescence), from each cell towards the mean position of all cells that
    # lie a little ahead of it in the measured cell-cycle position (circular, (0, window])
    pos = np.asarray(S.obs[a.cycle_key], dtype=float)
    E = np.asarray(S.obsm["X_umap"], dtype=float)
    cyc_emb = np.zeros_like(E)
    for i in range(S.n_obs):
        d = np.mod(pos - pos[i], 1.0)
        j = (d > 0) & (d <= a.cycle_window)
        if j.any():
            cyc_emb[i] = E[j].mean(0) - E[i]
    fields["labelling"] = ref

# ---- project every field into UMAP on the same graph ----
T = S[:, genes].copy()
allv = dict(fields)
allv.update({"ref_" + k: v for k, v in refs.items()})
emb = {}
for name, V in allv.items():
    T.layers[name] = V
    T.var[name + "_genes"] = True
    scv.tl.velocity_graph(T, vkey=name, xkey="Ms", n_jobs=a.jobs, show_progress_bar=False)
    scv.tl.velocity_embedding(T, vkey=name, basis="umap")
    emb[name] = np.asarray(T.obsm["%s_umap" % name])
if cyc_emb is not None:
    emb["ref_cycle"] = cyc_emb
    refs["cycle"] = None  # reference exists in the measured embedding only

rng = np.random.default_rng(910)
B = rng.integers(0, S.n_obs, size=(a.boot, S.n_obs))
per_cell = {}
rows = []
for rname, RV in refs.items():
    for name, V in fields.items():
        if name == "labelling" and rname == "labelling":
            continue
        pairs = [("umap", rowcos(emb[name], emb["ref_" + rname]))]
        if RV is not None:
            pairs.insert(0, ("gene", rowcos(V, RV)))
        for space, c in pairs:
            per_cell[(rname, name, space)] = c
            bm = np.array([np.nanmean(c[b]) for b in B])
            rows.append(dict(reference=rname, field=name, space=space, mean=np.nanmean(c),
                             lo=np.percentile(bm, 2.5), hi=np.percentile(bm, 97.5)))
res = pd.DataFrame(rows)
for rname, space in [(r, sp_) for r in refs for sp_ in ("gene", "umap") if (r, sp_) != ("cycle", "gene")]:
    vel = res[(res.reference == rname) & (res.space == space) &
              res.field.isin(["steady_state", "stochastic", "dynamical"])]
    best = vel.sort_values("mean").field.iloc[-1]
    for name in fields:
        if (rname, name, space) not in per_cell:
            continue
        d = per_cell[(rname, name, space)] - per_cell[(rname, best, space)]
        bm = np.array([np.nanmean(d[b]) for b in B])
        k = (res.reference == rname) & (res.field == name) & (res.space == space)
        res.loc[k, "vs_best_velocity"] = best
        res.loc[k, "diff"] = np.nanmean(d)
        res.loc[k, "diff_lo"] = np.percentile(bm, 2.5)
        res.loc[k, "diff_hi"] = np.percentile(bm, 97.5)
res["n_cells"] = S.n_obs
res["n_genes"] = len(genes)
res.to_csv(a.out + ".csv", index=False)
np.savez_compressed(a.out + "_emb.npz", umap=np.asarray(S.obsm["X_umap"]),
                    cell_type=np.asarray(S.obs.get("cell_type", pd.Series([""] * S.n_obs)).astype(str)),
                    **{k: v for k, v in emb.items()})
print(res.round(3).to_string())
say("done")
