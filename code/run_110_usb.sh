#!/bin/sh
# USB under this analysis' protocol (110_usb.py): per data set, the fate run (delta chosen from
# {10, 15} by fit) and then, for the three-day data sets, the held-out time point with that delta.
# Usage: sh run_110_usb.sh <gpu_larry> <gpu_ctm> <gpu_ctr>
cd "$(dirname "$0")"
mkdir -p ../out/heldout
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=../ext/env_usb/bin/python
delta_of() { $PY -c "import json; print(json.load(open('$1/snap_usb_run_log.json'))['delta'])"; }
chain() {  # gpu h5ad featdir tag heldout(0/1)
  G=$1; H=$2; FD=$3; T=$4
  CUDA_VISIBLE_DEVICES=$G $PY 110_usb.py fate $H $FD > ../logs/110_usb_fate_$T.log 2>&1; echo exit $? >> ../logs/110_usb_fate_$T.log
  if [ "$5" = 1 ] && [ -f $FD/snap_usb_run_log.json ]; then
    D=$(delta_of $FD)
    CUDA_VISIBLE_DEVICES=$G $PY 110_usb.py heldout $H $FD $T ../out/heldout/${T}_usb.csv --delta $D > ../logs/110_usb_heldout_$T.log 2>&1; echo exit $? >> ../logs/110_usb_heldout_$T.log
  fi
  touch ../logs/110_usb_$T.done
}
chain $1 ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all larry 1 &
chain $2 ../data/ctmulti/hsc_rna.h5ad ../out/feat_ctm ctm 0 &
chain $3 ../data/ctmulti/reprog_rna.h5ad ../out/feat_ctr ctr 1 &
wait
touch ../logs/110_usb_all.done
