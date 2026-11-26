#!/bin/sh
# CellRank's real-time kernel (moscot couplings, unbalanced tau_a 0.95, as rtk_ub on LARRY)
# on the two CellTag-multi data sets, then rescoring with 97.
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false
P=../env_ot/bin/python
$P 41_ot_fate.py ../data/ctmulti/hsc_rna.h5ad ../out/feat_ctm --method moscot --tag rtk_ub --verify-cells 0 --realtime --tau-a 0.95 > ../logs/ctm_rtk.log 2>&1; echo "exit $?" >> ../logs/ctm_rtk.log
$P 41_ot_fate.py ../data/ctmulti/reprog_rna.h5ad ../out/feat_ctr --method moscot --tag rtk_ub --verify-cells 0 --realtime --tau-a 0.95 > ../logs/ctr_rtk.log 2>&1; echo "exit $?" >> ../logs/ctr_rtk.log
touch ../logs/rtk.done
