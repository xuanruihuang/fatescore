#!/bin/bash
# Reproduce every figure, table and in-text number of the ICML submission
# "Does Agreement with Clonal Fate Measure Inferred Cell Dynamics?" (icml/main.tex).
#
#   bash reproduce.sh [ROOT]
#
# ROOT defaults to the folder holding this script. Layout under ROOT (the same layout the
# scripts in code/ assume, e.g. code/../data):
#   code/ tool/          the released code
#   data/                inputs (see DATA SOURCES below)
#   out/                 every result file, figure and table source
#   logs/                one log per stage (logs/<STAGE>.log)
#   env/ env_ot/ env_dyn/ ext/env_stb/ ext/env_cospar/   python environments (see REPRODUCE.md)
#   ext/scTimeBench/     the scTimeBench repository (cloned by the user, see below)
#
# Every stage can be skipped with SKIP_<STAGE>=1 (e.g. SKIP_STB_ARMS=1). Optional stages are off
# unless RUN_<STAGE>=1: PRESCIENT_REPRO (Appendix A reproduction). Stages that cannot be reconstructed from the
# scripts' docstrings or the README are written as commented TODOs.
#
# Other settings: PY, OT_PY, DYN_PY, STB_PY, COSPAR_PY (interpreters), GPU (CUDA device index,
# default 0), MT (threads for method stages, default 8).
#
# DATA SOURCES (place under ROOT/data; only items 1 and 2 are fetched here, as reproduce.sh does)
#  1. data/larry_invitro_adata_sub_raw.h5ad -- LARRY in vitro with spliced/unspliced layers,
#     figshare article 20780344, file 37028569 (https://ndownloader.figshare.com/files/37028569),
#     md5 d618305c0ee0e41570b1147e63df332a, 1,525,338,690 bytes. FETCHED.
#  2. data/celltagmulti/hsc_master_v2.csv -- CellTag-multi haematopoiesis clone table,
#     github.com/morris-lab/CellTag-multi-2023 clone_tables/hsc.rna&atac.r1&2_master_v2.csv,
#     md5 259a3ac68b0b3bfce20f166243fdf54c. FETCHED.
#  3. data/celltagmulti/b4_master_rev1_annotated.csv -- CellTag-multi reprogramming clone table,
#     same repository, clone_tables/b4.rna&atac.r1&2_master_rev1_annotated.csv (md5 not recorded).
#     Presence checked only.
#  4. data/GSE216521_RAW.tar -- GEO GSE216521 supplementary archive (68 CellRanger filtered h5
#     matrices, about 4.0 GB): https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE216521&format=file
#     Used members: GSM6681127 (day 2.5), GSM6681128-35 (day 5) for haematopoiesis;
#     GSM*_B4D{3,12,21}-RNA-r*-*_filtered_feature_bc_matrix.h5 for reprogramming (md5 not recorded).
#     Presence checked only.
#  5. data/GSM4185642/ -- Weinreb et al. 2020 in vitro, GEO GSM4185642,
#     https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4185nnn/GSM4185642/suppl/ :
#     GSM4185642_stateFate_inVitro_metadata.txt.gz (md5 0bb332b8538d58b536422804ee38ca77, recorded
#     at download), and for the optional PRESCIENT_REPRO stage also
#     GSM4185642_stateFate_inVitro_normed_counts.mtx.gz and _gene_names.txt.gz (md5 not recorded).
#     Presence checked only.
#  6. data/weinreb_fig5/ -- the files Weinreb et al. shared for their Fig. 5 (AllonKleinLab
#     paper-data; read as in scDiffEq/LARRY-dataset's WeinrebTestSet): timepoints.npy,
#     neutrophil_monocyte_trajectory_mask.npy, early_cells.npy, heldout_mask.npy,
#     clonal_fate_matrix.npy, coordinates_x.npy, coordinates_y.npy, PBA_predictions.npy,
#     FateID_predictions.npy, WOT_predictions.npy, smoothed_groundtruth_from_heldout.npy,
#     FINAL_fate_names.txt (md5 not recorded). Presence checked only.
#  7. data/labelling/ -- metabolic labelling (dynamo sample data, Qiu et al. 2022; scEU-seq,
#     Battich et al. 2020): hsc_small.h5ad (spliced/unspliced/labelling counts of
#     `hematopoiesis_raw`), hsc_published_velocity.h5ad (layer velocity_alpha_minus_gamma_s of
#     `hematopoiesis`, with time_<t>_gamma and X_umap), rpe1_small.h5ad (scEU-seq RPE1),
#     organoid_small.h5ad (scEU-seq intestinal organoid, with rotated_umap1/2). The
#     extraction into these files is not scripted in code/ (TODO). Organoid input: not recorded.
#     Presence checked only.
#  8. data/kegg/mm_birth_kegg_mmu04110.csv, mm_death_kegg_mmu04210.csv -- KEGG REST gene lists
#     (mouse cell cycle / apoptosis, column gene_symbol); optional PRESCIENT_REPRO stage only; the
#     download into csv is not scripted in code/ (TODO). Presence checked only.
#  Software fetched by the install scripts (network): scTimeBench must be cloned by the user into
#  ext/scTimeBench (github.com/li-lab-mcgill/scTimeBench; the commit used was not recorded);
#  MIOFlow commit 1c6ceb49841b17d178d4795439182b57250b467d (install_mioflow_old.sh); CoSpar from
#  PyPI (run_107_install.sh). 91b_gene_symbols.py queries mygene.info.
set -euo pipefail

ROOT=$(cd "${1:-$(dirname "$0")}" && pwd)
CODE=$ROOT/code
DATA=$ROOT/data
OUT=$ROOT/out
LOG=$ROOT/logs
PY=${PY:-$ROOT/env/bin/python}
OT_PY=${OT_PY:-$ROOT/env_ot/bin/python}
DYN_PY=${DYN_PY:-$ROOT/env_dyn/bin/python}
STB_PY=${STB_PY:-$ROOT/ext/env_stb/bin/python}
COSPAR_PY=${COSPAR_PY:-$ROOT/ext/env_cospar/bin/python}
GPU=${GPU:-0}
MT=${MT:-8}

H5=$DATA/larry_invitro_adata_sub_raw.h5ad
CTM_H5=$DATA/ctmulti/hsc_rna.h5ad
CTR_H5=$DATA/ctmulti/reprog_rna.h5ad
LAB=$OUT/d2_labels.tsv
FALL=$OUT/feat_all
FD2=$OUT/feat_d2
FCTM=$OUT/feat_ctm
FCTR=$OUT/feat_ctr
FIG5=$DATA/weinreb_fig5
META=$DATA/GSM4185642/GSM4185642_stateFate_inVitro_metadata.txt.gz
CTM_MASTER=$DATA/celltagmulti/hsc_master_v2.csv
CTR_MASTER=$DATA/celltagmulti/b4_master_rev1_annotated.csv
GEO_TAR=$DATA/GSE216521_RAW.tar

mkdir -p "$OUT" "$LOG"
cd "$ROOT"
# analysis default as in reproduce.sh; method stages raise it with mt()
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$CODE

mt () { ( export OMP_NUM_THREADS=$MT MKL_NUM_THREADS=$MT OPENBLAS_NUM_THREADS=$MT; "$@" ); }
gpu () { ( export CUDA_VISIBLE_DEVICES=$GPU; "$@" ); }
need () {  # need <path> <where to get it>
  if [ ! -e "$1" ]; then
    echo "missing: $1" >&2
    echo "  get it from: $2" >&2
    echo "  (or skip the stage that needs it with SKIP_<STAGE>=1)" >&2
    exit 1
  fi
}
# Run an install script from code/ (any proxy exports dropped).
portable () {
  mkdir -p "$ROOT/ext/tmp"
  sed -e '/_proxy=/d' "$1" | (cd "$CODE" && sh -s)
}
run_stage () {  # run_stage <NAME> <description> <function> [optional]
  local name=$1 desc=$2 fn=$3 opt=${4:-}
  local sv="SKIP_$name" rv="RUN_$name"
  if [ -n "$opt" ] && [ "${!rv:-0}" != 1 ]; then
    echo "== $name: skipped (optional; RUN_$name=1 to run)"; return 0
  fi
  if [ "${!sv:-0}" = 1 ]; then echo "== $name: skipped (SKIP_$name=1)"; return 0; fi
  echo "== $name: $desc"
  "$fn" 2>&1 | tee "$LOG/$name.log"
}

# ---------------------------------------------------------------------------------------------
# FETCH: the two inputs reproduce.sh fetches; everything else is checked by the stage needing it.
s_fetch () {
  if [ ! -f "$H5" ]; then
    mkdir -p "$(dirname "$H5")"
    curl -sSL -o "$H5" https://ndownloader.figshare.com/files/37028569
  fi
  echo "d618305c0ee0e41570b1147e63df332a  $H5" | md5sum -c -
  mkdir -p "$(dirname "$CTM_MASTER")"
  [ -s "$CTM_MASTER" ] || curl -sSL -o "$CTM_MASTER" \
    'https://raw.githubusercontent.com/morris-lab/CellTag-multi-2023/HEAD/clone_tables/hsc.rna%26atac.r1%262_master_v2.csv'
  echo "259a3ac68b0b3bfce20f166243fdf54c  $CTM_MASTER" | md5sum -c - \
    || echo "warning: the clone table has changed since 2026-11-06"
}

# LARRY feature directories and labels (as reproduce.sh): graph modes d2 and all; CellRank's
# velocity (cr_velo) and pseudotime (cr_pseudo) kernels.
s_larry_features () {
  $PY -s "$CODE/02b_labels.py" "$H5" "$LAB"
  for M in d2 all; do
    $PY -s "$CODE/02_features.py" "$H5" "$OUT/feat_$M" "$M"
    $PY -s "$CODE/02c_cellrank.py" "$OUT/feat_$M"
  done
}

# Section 4, own expression: day-2 expression classifier (AUROC 0.870) and sorted population alone
# (0.625), clone-disjoint cross-validation -> out/ev_d2_summary.csv (as reproduce.sh).
s_larry_expr_ref () {
  $PY -s "$CODE/03_evaluate.py" "$FD2" "$LAB" "$OUT/ev_d2"
}

# Connectivity kernel at 18 graph settings on LARRY (feat_all/cr_conn_n*_p*_d*.npy); the kernel on
# day-2 cells only (Section 4, 0.76-0.78) and its comparison table (README commands).
s_larry_kernels () {
  mt $PY -s "$CODE/29_expression_kernel.py" "$H5" "$FALL" --jobs 4
  mt $PY -s "$CODE/29_expression_kernel.py" "$H5" "$FD2" --d2-only --jobs 4
  $PY -s "$CODE/51_fate_share.py" "$FD2"
  $PY -s "$CODE/67_reference_intervals.py" "$FD2" "$LAB" "$OUT/refint_d2" --methods nm_cr_velo,nm_cr_pseudo
}

# CellTag-multi haematopoiesis (95) and reprogramming (95b): objects and feature directories.
s_ctm_build () {
  need "$GEO_TAR" "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE216521&format=file (GSE216521_RAW.tar)"
  need "$CTM_MASTER" "FETCH stage"
  $PY -s "$CODE/95_ctmulti_build.py" "$GEO_TAR" "$CTM_MASTER" "$DATA/ctmulti" "$FCTM"
}
s_ctr_build () {
  need "$GEO_TAR" "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE216521&format=file (GSE216521_RAW.tar)"
  need "$CTR_MASTER" "https://raw.githubusercontent.com/morris-lab/CellTag-multi-2023/HEAD/clone_tables/b4.rna%26atac.r1%262_master_rev1_annotated.csv"
  $PY -s "$CODE/95b_ctmulti_reprog_build.py" "$GEO_TAR" "$CTR_MASTER" "$DATA/ctmulti" "$FCTR"
}

# Connectivity kernel (18 settings; all cells, early cells) and CellRank's pseudotime kernel on the
# two CellTag-multi data sets. Reprogramming has no early-only mode: its day-3 cells carry only
# the annotation "early", so no terminal states exist among them.
s_ct_kernels () {
  mt $PY -s "$CODE/96_conn_kernel.py" "$CTM_H5" "$FCTM" --modes all,early,pseudotime
  mt $PY -s "$CODE/96_conn_kernel.py" "$CTR_H5" "$FCTR" --modes all,pseudotime
}

# Optimal transport (StationaryOT, Waddington-OT, moscot; script defaults) and CellRank's
# real-time kernel (moscot couplings, unbalanced tau_a 0.95) on the three data sets.
# Reprogramming StationaryOT at the doubled tolerance (Appendix A; run_ctr_statot2.sh).
s_ot_arms () {
  for M in statot wot moscot; do mt $OT_PY "$CODE/41_ot_fate.py" "$H5" "$FALL" --method $M; done
  mt $OT_PY "$CODE/41_ot_fate.py" "$H5" "$FALL" --method moscot --tag rtk_ub --verify-cells 0 --realtime --tau-a 0.95
  for M in statot wot moscot; do mt $OT_PY "$CODE/41_ot_fate.py" "$CTM_H5" "$FCTM" --method $M; done
  for M in wot moscot; do mt $OT_PY "$CODE/41_ot_fate.py" "$CTR_H5" "$FCTR" --method $M; done
  mt $OT_PY "$CODE/41_ot_fate.py" "$CTR_H5" "$FCTR" --method statot --statot-tol 2e-9
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  mt $OT_PY "$CODE/41_ot_fate.py" "$CTM_H5" "$FCTM" --method moscot --tag rtk_ub --verify-cells 0 --realtime --tau-a 0.95
  mt $OT_PY "$CODE/41_ot_fate.py" "$CTR_H5" "$FCTR" --method moscot --tag rtk_ub --verify-cells 0 --realtime --tau-a 0.95
}

# PRESCIENT (no growth weights; snap_prescient.npy) on the three data sets. The LARRY run
# directory is reused by the optional PRESCIENT_REPRO stage.
s_prescient_arms () {
  gpu mt $OT_PY "$CODE/43_prescient_fate.py" "$H5" "$FALL" --out-dir "$OUT/prescient_larry" --device cuda
  gpu mt $OT_PY "$CODE/43_prescient_fate.py" "$CTM_H5" "$FCTM" --out-dir "$OUT/prescient_ctm" --device cuda
  gpu mt $OT_PY "$CODE/43_prescient_fate.py" "$CTR_H5" "$FCTR" --out-dir "$OUT/prescient_ctr" --device cuda
}

# Flow matching (OT-CFM, SB-CFM; torchcfm), the setting chosen by fit: global scale, 50,000 steps,
# three seeds -> snap_otcfm_g50k.npy, snap_sbcfm_g50k.npy (fit in the json logs; Table 4).
s_flow_matching () {
  for D in "$H5 $FALL" "$CTM_H5 $FCTM" "$CTR_H5 $FCTR"; do
    set -- $D
    gpu mt $OT_PY "$CODE/94_flow_matching_fate.py" "$1" "$2" --tag _g50k --scale global --steps 50000 --device cuda:0
  done
}

# scTimeBench environment: one torch 2.8 / CUDA 12.8 environment for scNODE, scIMF, PI-SDE and
# MIOFlow, then MIOFlow pinned to commit 1c6ceb4 (Appendix A).
s_stb_install () {
  need "$ROOT/ext/scTimeBench" "git clone https://github.com/li-lab-mcgill/scTimeBench $ROOT/ext/scTimeBench (commit used not recorded)"
  need "$OT_PY" "env_ot (REPRODUCE.md): install_sctimebench.sh builds env_stb on its interpreter"
  portable "$CODE/install_sctimebench.sh"
  portable "$CODE/install_mioflow_old.sh"
}

# The four scTimeBench methods on the three data sets (runner defaults; each early cell carried
# forward by the trained model) -> snap_<method>.npy, stb_<method>_log.json (fit; Table 4).
s_stb_arms () {
  for D in "$H5 $FALL" "$CTM_H5 $FCTM" "$CTR_H5 $FCTR"; do
    set -- $D
    for M in scNODE scIMF PISDE MIOFlow; do
      gpu mt $STB_PY "$CODE/100_sctimebench_fate.py" "$1" "$2" $M
    done
  done
  # TODO 100_sctimebench_fate.py: the scIMF 1,000-epoch variant on haematopoiesis (Appendix B,
  # "16.4 both") was run with --set/--tag overrides that are not recorded.
}

# CellRank velocity + connectivity kernel 0.8/0.2 on LARRY (cr_vkck.npy, nm_cr_vkck.npy).
# (cr_velo and cr_pseudo come from LARRY_FEATURES; the steady-state velocity kernel from LADDER.)
s_cellrank_arms () {
  mt $PY -s "$CODE/70_combined_kernel.py" "$H5" "$FALL"
}

# Table 2, clonal-fate row (and the steady-state velocity kernel of Figure 2): the velocity kernel
# driven by dynamical / steady-state velocity, shuffled unspliced, degradation only, no field.
s_ladder () {
  mt $PY -s "$CODE/92_ladder_kernel.py" "$H5" "$FALL" "$FALL"
  $PY -s "$CODE/66_unsupervised_reference.py" "$FALL" "$LAB" "$OUT/ladder_fate.csv" \
    nm_cr_lad_velocity_dyn,nm_cr_lad_velocity_ss,nm_cr_lad_shuffled,nm_cr_lad_degradation,nm_cr_lad_connectivity
}

# Scoring: LARRY neutrophil shares (51; read by 99, 101, 104); CellTag-multi scores (97).
s_score () {
  $PY -s "$CODE/51_fate_share.py" "$FALL"
  OMP_NUM_THREADS=4 $PY -s "$CODE/97_fate_score.py" "$FCTM" "$OUT/ctm_fate_scores.csv" --pos Mono --ref cr_conn_all_n30_p30_d1
  OMP_NUM_THREADS=4 $PY -s "$CODE/97_fate_score.py" "$FCTR" "$OUT/ctr_fate_scores.csv" --pos reprogramming --ref cr_conn_all_n30_p30_d1
}

# Section 3.1: the published LARRY benchmark on its 335 test cells, matched by barcode with the GEO
# metadata -> out/weinreb_test_scores.csv.
s_published_test () {
  need "$FIG5/clonal_fate_matrix.npy" "the Weinreb et al. Fig. 5 files (header, item 6)"
  need "$META" "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4185nnn/GSM4185642/suppl/"
  OMP_NUM_THREADS=4 $PY -s "$CODE/101_weinreb_test_set.py" "$FIG5" "$H5" "$FALL" "$LAB" "$OUT/weinreb_test_scores.csv" "$META"
}

# Figure 1 -> out/fig_published.pdf
s_fig1 () {
  $PY -s "$CODE/101b_fig_published.py" "$OUT/weinreb_test_scores.csv" "$OUT/fig_published.pdf"
}

# Section 5 simulation (98): main grid (Fig. 3a, the green band of Fig. 2), estimation-error axis
# (Fig. 3b) and uniform decay with fate-wrong dynamics (Fig. 3c). The grids of the last two are
# those of the reported tables (priming 0.1/0.2, noise 0-1; priming 0.2, delta -0.5/0/0.5,
# decay 0-2); the exact command lines were not recorded.
s_simulation () {
  mt $PY -s "$CODE/98_mechanism_sim.py" "$OUT/mechanism_sim.csv"
  mt $PY -s "$CODE/98_mechanism_sim.py" "$OUT/mechanism_sim_noise.csv" --primings 0.1,0.2 --deltas 0,0.5,1 --noises 0,0.25,0.5,0.75,1
  mt $PY -s "$CODE/98_mechanism_sim.py" "$OUT/mechanism_sim_wrong.csv" --primings 0.2 --deltas=-0.5,0,0.5 --decays 0,0.5,1,2
}

# Section 3.2: gain over the connectivity kernel, 39 method-data set pairs -> out/calibrated_table.csv
s_table_gain () {
  OMP_NUM_THREADS=4 $PY -s "$CODE/99_calibrated_table.py" "$FALL" "$LAB" "$OUT/ctm_fate_scores.csv" "$OUT/ctr_fate_scores.csv" "$OUT/calibrated_table.csv"
}

# Figure 2 -> out/fig_gain.pdf; prints the simulated band ("band <lo>-<hi>"), kept for RESOLUTION.
s_fig2 () {
  $PY -s "$CODE/99b_fig_gain.py" "$OUT/calibrated_table.csv" "$OUT/mechanism_sim.csv" "$OUT/fig_gain.pdf" \
    | tee "$OUT/fig_gain_band.txt"
}

# Minimum detectable differences and kernel spread per data set -> out/resolution.csv
s_resolution () {
  local band
  band=$(sed -n 's/^band \([0-9.]*\)-\([0-9.]*\);.*/\1 \2/p' "$OUT/fig_gain_band.txt")
  [ -n "$band" ] || { echo "no band in $OUT/fig_gain_band.txt; run FIG2 first" >&2; exit 1; }
  set -- $band
  $PY -s "$CODE/102_resolution.py" "$OUT/calibrated_table.csv" "$1" "$2" "$OUT/resolution.csv"
}

# Figure 3 -> out/fig_mechanism.pdf
s_fig3 () {
  $PY -s "$CODE/103_fig_mechanism.py" "$OUT/mechanism_sim.csv" "$OUT/mechanism_sim_noise.csv" \
    "$OUT/mechanism_sim_wrong.csv" "$OUT/resolution.csv" "$OUT/fig_mechanism.pdf"
}

# Section 3.2 increment over the kernel -> out/incremental.csv; check of the one positive pair
# (scNODE on haematopoiesis; printed to logs/INCREMENT.log).
s_increment () {
  OMP_NUM_THREADS=4 $PY -s "$CODE/104_incremental.py" "$FALL" "$LAB" "$FCTM" "$FCTR" "$OUT/incremental.csv"
  $PY -s "$CODE/104b_nomass_check.py" "$FCTM" Mono snap_scnode
}

# Appendix A, PRESCIENT in three steps (optional; GPU, long): (i) the published read-out on our
# trained model; (ii) the published LARRY settings on our object, 5 seeds, no weights, and KEGG
# weights on our object; (iii) the authors' full data, 5 seeds, without and with KEGG weights.
s_prescient_repro () {
  need "$FIG5/clonal_fate_matrix.npy" "the Weinreb et al. Fig. 5 files (header, item 6)"
  need "$DATA/kegg/mm_birth_kegg_mmu04110.csv" "KEGG REST mmu04110 (header, item 8)"
  need "$DATA/kegg/mm_death_kegg_mmu04210.csv" "KEGG REST mmu04210 (header, item 8)"
  need "$DATA/GSM4185642/GSM4185642_stateFate_inVitro_normed_counts.mtx.gz" "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4185nnn/GSM4185642/suppl/"
  local P=$OUT/prescient_check
  mkdir -p "$P"
  $PY -s "$CODE/105_prescient_diag.py" "$FIG5" "$H5" "$FALL" "$LAB" snap_prescient
  mt $OT_PY "$CODE/105_prescient_resim.py" "$H5" "$OUT/prescient_larry" "$FALL" "$LAB" "$FIG5" snap_prescient "$P/resim_none.npz" --gpu "$GPU"
  for s in 1 2 3 4 5; do
    mt $OT_PY "$CODE/105_prescient_repro.py" "$H5" "$FALL" "$LAB" "$FIG5" "$P/repro_none_s$s" --seed $s --growth none --gpu "$GPU"
    mt $OT_PY "$CODE/105_prescient_repro.py" "$H5" "$FALL" "$LAB" "$FIG5" "$P/repro_kegg_s$s" --seed $s --growth kegg \
      --birth "$DATA/kegg/mm_birth_kegg_mmu04110.csv" --death "$DATA/kegg/mm_death_kegg_mmu04210.csv" --gpu "$GPU"
  done
  mt $OT_PY "$CODE/105_prescient_weinreb_prep.py" "$DATA/GSM4185642" "$DATA/kegg" "$P/full_data"
  for g in none kegg; do
    for s in 1 2 3 4 5; do
      mt $OT_PY "$CODE/105_prescient_full.py" "$P/full_data/data_full_$g.pt" "$H5" "$FALL" "$LAB" "$FIG5" "$P/full_${g}_s$s" \
        --growth $g --seed $s --gpu "$GPU"
    done
  done
  # mean +- SD over seeds: from the result_*_seed<k>.json files (no aggregation script).
}

# Section 6 and Tables 2/5: labelling velocity and the measured cell cycle (env_dyn).
s_labelling () {
  local L=$DATA/labelling
  need "$L/hsc_small.h5ad" "dynamo sample data hematopoiesis_raw (header, item 7)"
  need "$L/hsc_published_velocity.h5ad" "dynamo sample data hematopoiesis (header, item 7)"
  need "$L/rpe1_small.h5ad" "dynamo sample data scEU-seq RPE1 (header, item 7)"
  $DYN_PY "$CODE/91b_gene_symbols.py" "$L/hsc_small.h5ad" "$L/hsc_gene_symbols.tsv"
  mt $DYN_PY "$CODE/91_labelling_velocity.py" "$L/hsc_small.h5ad" "$OUT/labelling_hsc" --recipe published \
    --gene-map "$L/hsc_gene_symbols.tsv" --ref "$L/hsc_published_velocity.h5ad"
  $DYN_PY "$CODE/91b_gene_symbols.py" "$L/rpe1_small.h5ad" "$L/rpe1_gene_symbols.tsv"
  mt $DYN_PY "$CODE/91_labelling_velocity.py" "$L/rpe1_small.h5ad" "$OUT/labelling_rpe1" --recipe kin_tutorial --exp kin \
    --time-scale 60 --cycle-key Cell_cycle_relativePos --gene-map "$L/rpe1_gene_symbols.tsv" \
    --embed-cols RFP_log10_corrected,GFP_log10_corrected
  # scEU-seq intestinal organoid (Table 2 row "Labelling, organoid", Table 5 rows 5-6); the call
  # recorded in the run script of 2026-11-25
  need "$L/organoid_small.h5ad" "dynamo sample data scEU-seq organoid (header, item 7)"
  mt $DYN_PY "$CODE/91_labelling_velocity.py" "$L/organoid_small.h5ad" "$OUT/labelling_organoid" \
    --recipe organoid_tutorial --time-scale 60 --embed-cols rotated_umap1,rotated_umap2 --jobs 8
}

# Section 6, held-out time point (Waddington-OT's validation routine; 3,000 cells per day, three
# seeds): LARRY day 4 from days 2 and 6; reprogramming day 12 from days 3 and 21.
s_heldout () {
  mt $OT_PY "$CODE/90_ot_interpolation.py" "$H5" "$FALL" "$OUT/ot_interpolation.csv"
  mt $OT_PY "$CODE/90_ot_interpolation.py" "$CTR_H5" "$FCTR" "$OUT/ot_interpolation_ctr.csv"
}

# Table 3 (tab:heldout): held-out time point for the learned models (106), LARRY and reprogramming;
# share of the gap closed per seed -> out/heldout/summary.csv. The server ran one GPU per chain
# (run_106.sh); here the chains run one after another on GPU.
s_heldout_learned () {
  mkdir -p "$OUT/heldout"
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
  for D in "$H5 $FALL larry" "$CTR_H5 $FCTR ctr"; do
    set -- $D
    gpu $OT_PY "$CODE/106_heldout_refs.py" "$1" "$2" "$3" "$OUT/heldout/$3_refs.csv"
    gpu $OT_PY "$CODE/106_heldout_cfm.py" "$1" "$2" "$3" "$OUT/heldout/$3_cfm.csv"
    for M in scNODE scIMF PISDE MIOFlow; do
      gpu $STB_PY "$CODE/106_heldout_stb.py" "$1" "$2" "$3" $M "$OUT/heldout/$3_$M.csv"
    done
  done
  $PY -s "$CODE/106_heldout_summary.py" "$OUT/heldout" "$OUT/heldout/summary.csv"
}

# CoSpar state-only (107; optional, not cited as a scored method in the current main.tex).
s_cospar () {
  [ -x "$COSPAR_PY" ] || portable "$CODE/run_107_install.sh"
  for init in OT HighVar; do
    mt $COSPAR_PY "$CODE/107_cospar_state.py" "$H5" "$FALL" --larry --init $init
    mt $COSPAR_PY "$CODE/107_cospar_state.py" "$CTM_H5" "$FCTM" --init $init
    mt $COSPAR_PY "$CODE/107_cospar_state.py" "$CTR_H5" "$FCTR" --init $init
  done
  $PY -s "$CODE/107_cospar_score.py" "$FALL" "$LAB" "$FCTM" "$FCTR" "$OUT/cospar_scores.csv"
}

# Appendix C: the released scoring tool's self-check.
# USB and PACE (110): fate on the three data sets and the held-out time point. Their environments
# (ext/env_usb, ext/env_pace) and clones (USB commit 0cad29b, no licence; PACE commit 5f5a10e) are
# set up by hand as described in REPRODUCE.md. The runners take one GPU id per parallel chain.
s_usb_pace () {
  need "$ROOT/ext/env_usb/bin/python" "REPRODUCE.md, USB environment"
  need "$ROOT/ext/env_pace/bin/python" "REPRODUCE.md, PACE environment"
  sh "$CODE/run_110_usb.sh" "$GPU" "$GPU" "$GPU"
  sh "$CODE/run_110_pace.sh" "$GPU" "$GPU" "$GPU"
  sh "$CODE/run_110_pace_ipsc.sh" "$GPU" "$GPU" "$GPU"
  sh "$CODE/run_110_pace_heldout_ipsc.sh" "$GPU" "$GPU" "$GPU" "$GPU"
}

# PACE: the preset with the better distribution fit per data set becomes snap_pace_fit (110b).
s_pace_pick () {
  for d in "$FALL" "$FCTM" "$FCTR"; do
    need "$d/snap_pace.npy" "stage PACE (110_pace.py, run_110_pace.sh)"
    need "$d/snap_pace_ipsc.npy" "stage PACE (110_pace.py, run_110_pace_ipsc.sh)"
  done
  $PY -s "$CODE/110b_pace_fit_pick.py" "$OUT"
}

s_brier () {
  OMP_NUM_THREADS=4 $PY -s "$CODE/109_brier.py" "$FALL" "$LAB" "$FCTM" "$FCTR" "$OUT/brier.csv"
}

s_fig_concept () {
  local band
  band=$(sed -n 's/^band \([0-9.]*\)-\([0-9.]*\);.*/\1 \2/p' "$OUT/fig_gain_band.txt")
  [ -n "$band" ] || { echo "no band in $OUT/fig_gain_band.txt; run FIG2 first" >&2; exit 1; }
  set -- $band
  $PY -s "$CODE/113_fig_concept.py" "$OUT/calibrated_table.csv" "$1" "$2" "$OUT/fig_concept.pdf"
}

s_tool_check () {
  (cd "$ROOT/tool" && PYTHONPATH="$ROOT/tool" $PY -s tests/test_core.py)
}

# Figure 4: where the score comes from (115); classifier rows from LARRY_EXPR_REF's outputs.
s_fig_source () {
  need "$OUT/ev_d2_bootstrap.csv" "stage LARRY_EXPR_REF"
  $PY -s "$CODE/115_fig_source.py" "$OUT/ev_d2" "$FD2" "$FALL" "$LAB" "$OUT/calibrated_table.csv" \
    "$OUT/fig_source.pdf"
}

# Figure 6: held-out time point against the gain over the walk (114).
s_fig_heldout () {
  need "$OUT/heldout/summary_110.csv" "stages HELDOUT_LEARNED and USB_PACE"
  $PY -s "$CODE/114_fig_heldout_fate.py" "$OUT/calibrated_table.csv" "$OUT/heldout/summary_110.csv" \
    "$OUT/fig_heldout_fate.pdf"
}

# Appendix C: six terminal-state definitions (needs env_gpcca, code/86b_env_gpcca.sh) and the
# trifurcation and growth simulation (108).
s_terminal_sim () {
  need "$ROOT/env_gpcca/bin/python" "code/86b_env_gpcca.sh"
  sh "$CODE/run_108.sh"
}

# Appendix C: the walk plus a little direction, LARRY and haematopoiesis (111).
s_mix () {
  GPU=$GPU sh "$CODE/run_111.sh"
  sh "$CODE/run_111_ctm.sh"
}

# Appendix C: clones whose present state leaves fate open (112).
s_hard_cells () {
  sh "$CODE/run_112.sh"
}

# Appendix C: learned models rescored on clones whose cells all reach a scored fate (116).
s_readout () {
  $PY -s "$CODE/116_readout_check.py" "$OUT" "$OUT/readout_check.csv"
}

# Section 5: the learned-model pipeline (OT-CFM, carry forward, 20-NN read-out) on simulated
# snapshots, against the walk, the true-velocity kernel and a state classifier (119; one GPU).
s_learned_control () {
  gpu mt $OT_PY -s "$CODE/119_learned_control.py" "$OUT/learned_control119.csv" --device cuda:0
}

# Section 3.2: bootstrap p-values of the 48 gains and Holm's correction over them (118).
s_multiplicity () {
  OMP_NUM_THREADS=1 $PY -s "$CODE/118_multiplicity.py" "$FALL" "$LAB" "$FCTM" "$FCTR" \
    "$OUT/calibrated_table.csv" "$OUT/multiplicity118.csv"
}

# Appendix C: six fates on LARRY, macro one-versus-rest AUROC (117).
s_multifate () {
  OMP_NUM_THREADS=1 $PY -s "$CODE/117_multifate.py" "$FALL" "$LAB" "$OUT/multifate117_larry.csv"
}

# The six figure files main.tex includes (icml/figures/).
s_figures_copy () {
  mkdir -p "$ROOT/icml/figures"
  cp "$OUT/fig_concept.pdf" "$OUT/fig_published.pdf" "$OUT/fig_gain.pdf" "$OUT/fig_source.pdf" \
    "$OUT/fig_mechanism.pdf" "$OUT/fig_heldout_fate.pdf" "$ROOT/icml/figures/"
}

# ---------------------------------------------------------------------------------------------
run_stage FETCH            "LARRY deposit and CellTag-multi clone table"          s_fetch
run_stage LARRY_FEATURES   "LARRY feature directories, labels, CellRank kernels"  s_larry_features
run_stage LARRY_EXPR_REF   "day-2 expression classifier (Section 4)"              s_larry_expr_ref
run_stage LARRY_KERNELS    "connectivity kernel, 18 settings; day-2-only graph"   s_larry_kernels
run_stage CTM_BUILD        "CellTag-multi haematopoiesis (95)"                    s_ctm_build
run_stage CTR_BUILD        "CellTag-multi reprogramming (95b)"                    s_ctr_build
run_stage CT_KERNELS       "CellTag-multi kernels and pseudotime kernel (96)"     s_ct_kernels
run_stage OT_ARMS          "optimal transport and real-time kernel (41)"          s_ot_arms
run_stage PRESCIENT_ARMS   "PRESCIENT (43)"                                       s_prescient_arms
run_stage FLOW_MATCHING    "OT-CFM and SB-CFM (94)"                               s_flow_matching
run_stage STB_INSTALL      "scTimeBench environment"                              s_stb_install
run_stage STB_ARMS         "scNODE, scIMF, PI-SDE, MIOFlow (100)"                 s_stb_arms
run_stage CELLRANK_ARMS    "velocity + connectivity kernel (70)"                  s_cellrank_arms
run_stage LADDER           "deconstruction ladder on clonal fate (92, 66)"        s_ladder
run_stage COSPAR           "CoSpar state-only (107)"                              s_cospar
run_stage USB_PACE         "USB and PACE (110)"                                   s_usb_pace
run_stage PACE_PICK        "PACE preset chosen by fit (110b)"                     s_pace_pick
run_stage SCORE            "fate shares (51) and CellTag-multi scores (97)"       s_score
run_stage PUBLISHED_TEST   "published 335-cell LARRY test set (101)"              s_published_test
run_stage FIG1             "Figure 2 (101b)"                                      s_fig1
run_stage SIMULATION       "simulated positive control (98)"                      s_simulation
run_stage LEARNED_CONTROL  "learned-model pipeline on simulated snapshots (119)"  s_learned_control
run_stage TABLE_GAIN       "gain over the kernel, three data sets (99)"           s_table_gain
run_stage FIG2             "Figure 3 (99b)"                                       s_fig2
run_stage RESOLUTION       "minimum detectable differences (102)"                 s_resolution
run_stage MULTIPLICITY     "Holm correction over the 48 gains (118)"              s_multiplicity
run_stage FIG3             "Figure 5 (103)"                                       s_fig3
run_stage INCREMENT        "increment over the kernel (104, 104b)"                s_increment
run_stage BRIER            "Brier score, raw and recalibrated (109)"              s_brier
run_stage FIG_CONCEPT      "Figure 1 (113)"                                       s_fig_concept
run_stage FIG_SOURCE       "Figure 4 (115)"                                       s_fig_source
run_stage READOUT          "learned models on fully reached clones (116)"         s_readout
run_stage MULTIFATE        "six fates on LARRY (117)"                             s_multifate
run_stage TERMINAL_SIM     "terminal states; trifurcation and growth (108)"       s_terminal_sim
run_stage MIX              "the walk plus a little direction (111)"               s_mix
run_stage HARD_CELLS       "clones whose state leaves fate open (112)"            s_hard_cells
run_stage PRESCIENT_REPRO  "PRESCIENT reproduction, Appendix A (105)"             s_prescient_repro optional
run_stage LABELLING        "metabolic labelling and cell cycle (91)"              s_labelling
run_stage HELDOUT          "held-out time point (90)"                             s_heldout
run_stage HELDOUT_LEARNED  "held-out time point, learned models (106)"            s_heldout_learned
run_stage FIG_HELDOUT      "Figure 6 (114)"                                       s_fig_heldout
run_stage TOOL_CHECK       "scoring tool self-check"                              s_tool_check
run_stage FIGURES_COPY     "figures into icml/figures"                            s_figures_copy
echo "done; results in $OUT, logs in $LOG, figures in $ROOT/icml/figures"
