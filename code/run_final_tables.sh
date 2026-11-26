#!/bin/sh
# After run_rescore_all.sh: the 335-cell published test set (101), its figure (101b),
# the cross-data-set calibrated table (99) and the gain figure (99b).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
while [ ! -f ../logs/rescore3.done ]; do sleep 60; done
../env/bin/python 101_weinreb_test_set.py ../data/weinreb_fig5 ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all ../out/d2_labels.tsv ../out/weinreb_test_scores.csv > ../logs/101_weinreb_final.log 2>&1; echo exit $? >> ../logs/101_weinreb_final.log
../env/bin/python 101b_fig_published.py ../out/weinreb_test_scores.csv ../out/fig_published.pdf > ../logs/101b.log 2>&1; echo exit $? >> ../logs/101b.log
../env/bin/python 99_calibrated_table.py ../out/feat_all ../out/d2_labels.tsv ../out/ctm_fate_scores.csv ../out/ctr_fate_scores.csv ../out/calibrated_table.csv > ../logs/99_final.log 2>&1; echo exit $? >> ../logs/99_final.log
../env/bin/python 99b_fig_gain.py ../out/calibrated_table.csv ../out/mechanism_sim.csv ../out/fig_gain.pdf > ../logs/99b_final.log 2>&1; echo exit $? >> ../logs/99b_final.log
touch ../logs/final_tables.done
