#!/bin/sh
# PACE under this analysis' protocol (110_pace.py, preset op_cite_pace_ode_100d): fate runs on the
# three data sets, held-out time point on LARRY and reprogramming (3 seeds).
# Usage: sh run_110_pace.sh <gpu_a> <gpu_b> <gpu_c>
cd "$(dirname "$0")"
mkdir -p ../out/heldout ../out/pace110
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=../ext/env_pace/bin/python
L="../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all larry"
M="../data/ctmulti/hsc_rna.h5ad ../out/feat_ctm ctm"
R="../data/ctmulti/reprog_rna.h5ad ../out/feat_ctr ctr"
fate() { set -- $2; CUDA_VISIBLE_DEVICES=$G $PY 110_pace.py fate $1 $2 $3 ../out/pace110/$3 > ../logs/110_pace_fate_$3.log 2>&1; echo exit $? >> ../logs/110_pace_fate_$3.log; }
held() { set -- $2; CUDA_VISIBLE_DEVICES=$G $PY 110_pace.py heldout $1 $2 $3 ../out/pace110/$3 ../out/heldout/$3_pace.csv > ../logs/110_pace_heldout_$3.log 2>&1; echo exit $? >> ../logs/110_pace_heldout_$3.log; }
(G=$1; fate x "$L"; held x "$L"; touch ../logs/110_pace_a.done) &
(G=$2; fate x "$M"; held x "$R"; touch ../logs/110_pace_b.done) &
(G=$3; fate x "$R"; touch ../logs/110_pace_c.done) &
wait
touch ../logs/110_pace_all.done
