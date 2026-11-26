#!/bin/sh
# Held-out time point for PACE with the preset chosen by fate fit on LARRY and reprogramming
# (ipsc_liu_pace_ode_50d); one process per LARRY seed, reprogramming seeds in sequence.
# Usage: sh run_110_pace_heldout_ipsc.sh <gpu_l0> <gpu_l1> <gpu_l2> <gpu_ctr>
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=../ext/env_pace/bin/python
L="../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all larry"
for s in 0 1 2; do
  eval G=\${$((s + 1))}
  (CUDA_VISIBLE_DEVICES=$G $PY 110_pace.py heldout $L ../out/pace110/larry ../out/heldout/larry_pace_ipsc_s$s.csv --experiment ipsc_liu_pace_ode_50d --name snap_pace_ipsc --seed-start $s --seeds 1 > ../logs/110_pace_ipsc_heldout_larry_s$s.log 2>&1; echo exit $? >> ../logs/110_pace_ipsc_heldout_larry_s$s.log) &
done
(CUDA_VISIBLE_DEVICES=$4 $PY 110_pace.py heldout ../data/ctmulti/reprog_rna.h5ad ../out/feat_ctr ctr ../out/pace110/ctr ../out/heldout/ctr_pace_ipsc.csv --experiment ipsc_liu_pace_ode_50d --name snap_pace_ipsc > ../logs/110_pace_ipsc_heldout_ctr.log 2>&1; echo exit $? >> ../logs/110_pace_ipsc_heldout_ctr.log) &
wait
touch ../logs/110_pace_ipsc_heldout.done
