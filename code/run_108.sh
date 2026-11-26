#!/bin/sh
# Preliminary checks after an LLM review: terminal-state sensitivity (108_terminal_alt.py, in
# env_gpcca for the GPCCA step) and the trifurcation/growth simulation (108_mechanism_multi.py).
# Run from code/. Two independent chains; each writes its log and a .done marker.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
( ../env_gpcca/bin/python 108_terminal_alt.py ../data/larry_invitro_adata_sub_raw.h5ad ../out/feat_all \
    ../out/d2_labels.tsv ../out/terminal_states_alt.csv > ../logs/108_terminal.log 2>&1
  echo exit $? >> ../logs/108_terminal.log; touch ../logs/108_terminal.done ) &
( ../env/bin/python 108_mechanism_multi.py ../out/mechanism_multi.csv > ../logs/108_sim.log 2>&1
  echo exit $? >> ../logs/108_sim.log
  ../env/bin/python 108_mechanism_multi.py ../out/mechanism_multi_noise.csv --primings 0.2 \
    --deltas 0.5,1 --noises 0.5,1 > ../logs/108_sim_noise.log 2>&1
  echo exit $? >> ../logs/108_sim_noise.log; touch ../logs/108_sim.done ) &
wait
