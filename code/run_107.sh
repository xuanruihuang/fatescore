#!/bin/sh
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}   # repository root
# CoSpar state-only on the three lineage data sets (OT initialisation, the default), in parallel;
# then the HighVar initialisation; then scoring (107_cospar_score.py, analysis environment).
cd $ROOT/code
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
PY=../ext/env_cospar/bin/python
run() { $PY 107_cospar_state.py "$1" "$2" $3 > ../logs/107_$4.log 2>&1; echo "exit $?" >> ../logs/107_$4.log; }
run ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all "--larry" larry &
run ../data/ctmulti/hsc_rna.h5ad ../out/feat_ctm "" ctm &
run ../data/ctmulti/reprog_rna.h5ad ../out/feat_ctr "" ctr &
wait
run ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all "--larry --init HighVar" larry_hv &
run ../data/ctmulti/hsc_rna.h5ad ../out/feat_ctm "--init HighVar" ctm_hv &
run ../data/ctmulti/reprog_rna.h5ad ../out/feat_ctr "--init HighVar" ctr_hv &
wait
touch ../logs/107_maps.done
