#!/bin/sh
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}   # repository root
# StationaryOT on the reprogramming data set again: the first run (tolerance 1e-9 per cell) ended at a
# marginal error 1.08 times the tolerance after 3.5 h and saved nothing; tolerance doubled here.
cd $ROOT/code
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
../env_ot/bin/python 41_ot_fate.py ../data/ctmulti/reprog_rna.h5ad ../out/feat_ctr --method statot --statot-tol 2e-9 > ../logs/ctr_41_statot2.log 2>&1
echo "exit $?" >> ../logs/ctr_41_statot2.log
until [ -f ../logs/ctr_stb.done ]; do sleep 120; done
OMP_NUM_THREADS=4 ../env/bin/python 97_fate_score.py ../out/feat_ctr ../out/ctr_fate_scores.csv --pos reprogramming --ref cr_conn_all_n30_p30_d1 > ../logs/ctr_97_final2.log 2>&1
echo "exit $?" >> ../logs/ctr_97_final2.log
