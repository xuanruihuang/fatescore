#!/bin/sh
# 111, step 2: the real-time kernel family on CellTag-multi haematopoiesis (the family that won on LARRY).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
../env_ot/bin/python 111_mix_rtk_full.py ../data/ctmulti/hsc_rna.h5ad ../out/feat_ctm ../out/mix111_ctm > ../logs/111_ctm_rtk.log 2>&1
echo "exit $?" >> ../logs/111_ctm_rtk.log
../env/bin/python 111_mix_score_ct.py ../out/mix111_ctm ../out/feat_ctm Mono ../out/mix111_ctm_scores.csv > ../logs/111_ctm_score.log 2>&1
echo "exit $?" >> ../logs/111_ctm_score.log
touch ../logs/111_ctm.done
