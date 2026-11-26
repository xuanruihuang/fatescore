#!/bin/bash
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}   # repository root
# §16.22: PRESCIENT's sensitivity to the downsampling seed, as a descriptive
# result. Each seed rebuilds the in vivo object with the same design as
# §16.20 (21,000 cells per later timepoint, every early cell kept, labels
# from the full experiment) and runs only PRESCIENT and the expression
# baseline. PRESCIENT's own training seed stays at the package default, so
# the only thing that varies is which late cells were sampled.
set -u
cd $ROOT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
for s in 1 2; do
  echo "=== seed $s: build ==="
  env/bin/python code/42_build_arm.py --arm invivo \
      --data data/larry_arms --meta data/larry_invivo_meta --out out \
      --downsample-late 21000 --seed "$s" --suffix "_ds21000_s$s" \
      > "logs/build_invivo_ds_s$s.log" 2>&1 || { echo "build $s failed"; continue; }
  echo "=== seed $s: prescient ==="
  CUDA_VISIBLE_DEVICES=2 env_ot/bin/python code/43_prescient_fate.py \
      "out/larry_invivo_ds21000_s$s.h5ad" "out/feat_invivo_ds21000_s$s" \
      --out-dir "out/prescient_invivo_ds_s$s" --device cuda \
      > "logs/ds_prescient_s$s.log" 2>&1 || { echo "prescient $s failed"; continue; }
  echo "=== seed $s: score ==="
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  env/bin/python code/30_kernel_eval.py "out/feat_invivo_ds21000_s$s" \
      out/invivo_labels.tsv "out/otcmp_invivo_ds_s$s" \
      --prefixes snap_,expr_ --pca-prefixes expr_ --ref expr_hvg \
      --pos Ery --neg DC --pop-ref Lin-Kit+Sca1hi --n-jobs 12 \
      > "logs/ds_eval_s$s.log" 2>&1 || { echo "score $s failed"; continue; }
  echo "=== seed $s done ==="
done
echo "=== all seeds done ==="
