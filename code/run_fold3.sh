#!/bin/sh
# Fold CoSpar, USB and PACE into every table and figure: pick the fit-chosen PACE preset, rescore the
# CellTag-multi data sets (97), then 99, 99b, 101 (335 cells, barcode match), 101b, 102, 103, 104, 109, 113.
cd "$(dirname "$0")"
[ -f ../logs/fold3.lock ] && exit 0
touch ../logs/fold3.lock
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PY=../env/bin/python
L=../logs
$PY 110b_pace_fit_pick.py ../out > $L/fold3_pick.log 2>&1
$PY 97_fate_score.py ../out/feat_ctm ../out/ctm_fate_scores.csv --pos Mono --ref cr_conn_all_n30_p30_d1 > $L/fold3_97_ctm.log 2>&1
$PY 97_fate_score.py ../out/feat_ctr ../out/ctr_fate_scores.csv --pos reprogramming --ref cr_conn_all_n30_p30_d1 > $L/fold3_97_ctr.log 2>&1
$PY 99_calibrated_table.py ../out/feat_all ../out/d2_labels.tsv ../out/ctm_fate_scores.csv ../out/ctr_fate_scores.csv ../out/calibrated_table.csv > $L/fold3_99.log 2>&1
$PY 99b_fig_gain.py ../out/calibrated_table.csv ../out/mechanism_sim.csv ../out/fig_gain.pdf > $L/fold3_99b.log 2>&1
$PY 101_weinreb_test_set.py ../data/weinreb_fig5 ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all ../out/d2_labels.tsv ../out/weinreb_test_scores.csv ../ext/weinreb_full/GSM4185642_stateFate_inVitro_metadata.txt.gz > $L/fold3_101.log 2>&1
$PY 101b_fig_published.py ../out/weinreb_test_scores.csv ../out/fig_published.pdf > $L/fold3_101b.log 2>&1
$PY 102_resolution.py ../out/calibrated_table.csv 0.048 0.097 ../out/resolution.csv > $L/fold3_102.log 2>&1
$PY 103_fig_mechanism.py ../out/mechanism_sim.csv ../out/mechanism_sim_noise.csv ../out/mechanism_sim_wrong.csv ../out/resolution.csv ../out/fig_mechanism.pdf > $L/fold3_103.log 2>&1
$PY 104_incremental.py ../out/feat_all ../out/d2_labels.tsv ../out/feat_ctm ../out/feat_ctr ../out/incremental.csv > $L/fold3_104.log 2>&1
$PY 109_brier.py ../out/feat_all ../out/d2_labels.tsv ../out/feat_ctm ../out/feat_ctr ../out/brier.csv > $L/fold3_109.log 2>&1
$PY 113_fig_concept.py ../out/calibrated_table.csv 0.048 0.097 ../out/fig_concept.pdf > $L/fold3_113.log 2>&1
touch $L/fold3.done
