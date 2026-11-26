#!/bin/sh
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}   # repository root
# After the scTimeBench methods: rescore the three lineage data sets with every method present.
cd $ROOT/code
while [ ! -f ../logs/stbm_ctm_stb.done ] || [ ! -f ../logs/stbm_ctr_stb.done ] || [ ! -f ../logs/stbm_larry_stb.done ]; do sleep 60; done
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
../env/bin/python 97_fate_score.py ../out/feat_ctm ../out/ctm_fate_scores.csv --pos Mono --ref cr_conn_all_n30_p30_d1 > ../logs/ctm_97_final3.log 2>&1; echo "exit $?" >> ../logs/ctm_97_final3.log
../env/bin/python 97_fate_score.py ../out/feat_ctr ../out/ctr_fate_scores.csv --pos reprogramming --ref cr_conn_all_n30_p30_d1 > ../logs/ctr_97_final3.log 2>&1; echo "exit $?" >> ../logs/ctr_97_final3.log
../env/bin/python 51_fate_share.py ../out/feat_all > ../logs/larry_51_final3.log 2>&1; echo "exit $?" >> ../logs/larry_51_final3.log
touch ../logs/rescore3.done
