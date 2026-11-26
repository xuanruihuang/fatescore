#!/bin/bash
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}   # repository root
# Section 17.4: delete the one intermediate sampling day and see whether
# anything measurable is lost. Two readings, registered before they ran and
# not to be mixed: --keep-days filters before the PCA, so neither the
# representation nor the coupling chain sees day 4 (the experiment that never
# sampled it); --couple-days filters after the PCA, so the representation is
# the full one and only the coupling chain is shortened (the method declining
# to use a day that was measured). Every early cell survives either way --
# the script asserts it -- so the evaluation set does not move.
#
# StationaryOT is deliberately not ablated: it takes no sampling day at all,
# which is why it is this section's reference point rather than its subject.
set -u
cd $ROOT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
H5=data/larry_invitro_adata_sub_raw.h5ad
FD=out/feat_all

run () {   # method tag extra-args...
  local m="$1"; local t="$2"; shift 2
  echo "=== $m $t ==="
  env_ot/bin/python code/41_ot_fate.py "$H5" "$FD" \
      --method "$m" --tag "$t" --verify-cells 0 "$@" \
      > "logs/abl_$t.log" 2>&1 \
    && echo "=== $t done ===" || echo "=== $t FAILED ==="
}

run wot    wot_kd26    --keep-days   2,6 --wot-out out/wot_tmaps_kd26
run wot    wot_cd26    --couple-days 2,6 --wot-out out/wot_tmaps_cd26
run moscot moscot_kd26 --keep-days   2,6
run moscot moscot_cd26 --couple-days 2,6
echo "=== all ablation runs done ==="
