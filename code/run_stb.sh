#!/bin/sh
# The scTimeBench-wrapped methods on one lineage data set:
#   run_stb.sh <tag> <h5ad> <featdir> <gpu> <methods> [extra arguments to 100_sctimebench_fate.py]
TAG=$1; H5=$2; FD=$3; GPU=$4; METHODS=$5; shift 5; EXTRA="$*"
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for M in $(echo $METHODS | tr ',' ' '); do
  ../ext/env_stb/bin/python 100_sctimebench_fate.py $H5 $FD $M $EXTRA > ../logs/${TAG}_100_$M.log 2>&1
  echo "exit $?" >> ../logs/${TAG}_100_$M.log
done
touch ../logs/${TAG}_stb.done
