#!/bin/sh
# Hard-cell stratification (112_hard_cells.py) on LARRY, CellTag-multi haematopoiesis and reprogramming.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
../env/bin/python 112_hard_cells.py larry ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all ../out/d2_labels.tsv ../out/hard_cells_larry.csv > ../logs/112_larry.log 2>&1; echo exit $? >> ../logs/112_larry.log
../env/bin/python 112_hard_cells.py ct ../data/ctmulti/hsc_rna.h5ad ../out/feat_ctm Mono ../out/hard_cells_ctm.csv > ../logs/112_ctm.log 2>&1; echo exit $? >> ../logs/112_ctm.log
../env/bin/python 112_hard_cells.py ct ../data/ctmulti/reprog_rna.h5ad ../out/feat_ctr reprogramming ../out/hard_cells_ctr.csv > ../logs/112_ctr.log 2>&1; echo exit $? >> ../logs/112_ctr.log
touch ../logs/112.done
