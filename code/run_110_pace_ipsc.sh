#!/bin/sh
# PACE's second documented scRNA-PCA preset (ipsc_liu_pace_ode_50d) on the three data sets, fate
# runs only; per data set the preset with the better distribution fit is kept (110_pace.py).
# Usage: sh run_110_pace_ipsc.sh <gpu_larry> <gpu_ctm> <gpu_ctr>
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=../ext/env_pace/bin/python
run() { CUDA_VISIBLE_DEVICES=$1 $PY 110_pace.py fate $2 $3 $4 ../out/pace110/$4 --experiment ipsc_liu_pace_ode_50d --name snap_pace_ipsc > ../logs/110_pace_ipsc_fate_$4.log 2>&1; echo exit $? >> ../logs/110_pace_ipsc_fate_$4.log; }
run $1 ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all larry &
run $2 ../data/ctmulti/hsc_rna.h5ad ../out/feat_ctm ctm &
run $3 ../data/ctmulti/reprog_rna.h5ad ../out/feat_ctr ctr &
wait
touch ../logs/110_pace_ipsc.done
