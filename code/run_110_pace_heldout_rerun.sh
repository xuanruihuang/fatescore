#!/bin/sh
# Rerun of the PACE held-out seeds that ran out of GPU memory on a shared GPU (LARRY seed 1;
# reprogramming seeds 0-2), one process per seed. Usage: sh run_110_pace_heldout_rerun.sh <g1> <g2> <g3> <g4>
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=../ext/env_pace/bin/python
X="--experiment ipsc_liu_pace_ode_50d --name snap_pace_ipsc --seeds 1"
(CUDA_VISIBLE_DEVICES=$1 $PY 110_pace.py heldout ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all larry ../out/pace110/larry ../out/heldout/larry_pace_ipsc_s1.csv $X --seed-start 1 > ../logs/110_pace_ipsc_heldout_larry_s1.log 2>&1; echo exit $? >> ../logs/110_pace_ipsc_heldout_larry_s1.log) &
G2=$2; G3=$3; G4=$4
for s in 0 1 2; do
  eval G=\$G$((s + 2))
  (CUDA_VISIBLE_DEVICES=$G $PY 110_pace.py heldout ../data/ctmulti/reprog_rna.h5ad ../out/feat_ctr ctr ../out/pace110/ctr ../out/heldout/ctr_pace_ipsc_s$s.csv $X --seed-start $s > ../logs/110_pace_ipsc_heldout_ctr_s$s.log 2>&1; echo exit $? >> ../logs/110_pace_ipsc_heldout_ctr_s$s.log) &
done
wait
touch ../logs/110_pace_ipsc_heldout_rerun.done
