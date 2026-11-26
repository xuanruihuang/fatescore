#!/bin/sh
# Held-out time point for the learned models (106_heldout_*.py): LARRY (day 4 from days 2 and 6),
# then CellTag-multi reprogramming (day 12 from days 3 and 21). One chain per GPU.
# Usage: sh run_106.sh <gpu_refs_cfm> <gpu_scnode> <gpu_scimf> <gpu_pisde> <gpu_mioflow>
cd "$(dirname "$0")"
mkdir -p ../out/heldout
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
L="../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all larry"
R="../data/ctmulti/reprog_rna.h5ad ../out/feat_ctr ctr"
chain_ot() {
  for D in "$L" "$R"; do
    set -- $D
    CUDA_VISIBLE_DEVICES=$G ../env_ot/bin/python 106_heldout_refs.py $1 $2 $3 ../out/heldout/$3_refs.csv > ../logs/106_$3_refs.log 2>&1; echo exit $? >> ../logs/106_$3_refs.log
    CUDA_VISIBLE_DEVICES=$G ../env_ot/bin/python 106_heldout_cfm.py $1 $2 $3 ../out/heldout/$3_cfm.csv > ../logs/106_$3_cfm.log 2>&1; echo exit $? >> ../logs/106_$3_cfm.log
  done
  touch ../logs/106_ot.done
}
chain_stb() {
  for D in "$L" "$R"; do
    set -- $D
    CUDA_VISIBLE_DEVICES=$G ../ext/env_stb/bin/python 106_heldout_stb.py $1 $2 $3 $M ../out/heldout/$3_$M.csv > ../logs/106_$3_$M.log 2>&1; echo exit $? >> ../logs/106_$3_$M.log
  done
  touch ../logs/106_$M.done
}
G=$1 chain_ot &
G=$2 M=scNODE chain_stb &
G=$3 M=scIMF chain_stb &
G=$4 M=PISDE chain_stb &
G=$5 M=MIOFlow chain_stb &
wait
touch ../logs/106_all.done
