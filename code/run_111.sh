#!/bin/sh
# 111: mixtures of the connectivity kernel with a little direction, on LARRY.
# Velocity / pseudotime / debiased families (main env) and the real-time kernel family (env_ot)
# run in parallel; scoring waits for both.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=8 PYTHONUNBUFFERED=1
MIX=../out/mix111
mkdir -p $MIX
( ../env/bin/python 111_mix_velocity.py ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all $MIX > ../logs/111_velocity.log 2>&1; echo "exit $?" >> ../logs/111_velocity.log ) &
( CUDA_VISIBLE_DEVICES=${GPU:-7} OMP_NUM_THREADS=4 ../env_ot/bin/python 111_mix_rtk.py ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all $MIX > ../logs/111_rtk.log 2>&1; echo "exit $?" >> ../logs/111_rtk.log ) &
wait
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ../env/bin/python 111_mix_score.py $MIX ../out/feat_all ../out/d2_labels.tsv ../out/mix111_scores.csv > ../logs/111_score.log 2>&1
echo "exit $?" >> ../logs/111_score.log
touch ../logs/111.done
