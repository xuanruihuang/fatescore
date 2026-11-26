"""Direction scores in the benchmarks' own convention, for the figures.

71_direction_embedding.py and 84_benchmark_direction.py write two versions of
each score: pooled over all boundary cells, and averaged within each transition
and then over transitions (VeloAE's cross_boundary_correctness); ICCoh both on
the projected vectors and, as VeloAE's inner_cluster_coh, on the gene-space
vectors with the cell in its own neighbour list, averaged within clusters and
then over clusters. This writes the benchmark-convention version under the
column names the figure scripts read: cbdir_fwd, cbdir_rev, fwd_minus_rev,
diff_lo, diff_hi and, in the embedding, iccoh (gene space, NaN for fields
defined only in the embedding). In principal components iccoh stays on the
projected vectors.

Usage: 89_direction_convention.py <outdir>
"""
import sys

import numpy as np
import pandas as pd

R = sys.argv[1].rstrip("/") + "/"
for src, dst, emb in (("direction_embedding3.csv", "direction_embedding_bench.csv", "umap"),
                      ("benchmark_direction2.csv", "benchmark_direction_bench.csv", "embedding")):
    E = pd.read_csv(R + src)
    out = E.copy()
    out["cbdir_fwd"] = E["cbdir_fwd_t"]
    out["cbdir_rev"] = E["cbdir_rev_t"]
    out["fwd_minus_rev"] = E["fwd_minus_rev_t"]
    out["diff_lo"] = E["diff_t_lo"]
    out["diff_hi"] = E["diff_t_hi"]
    m = E["space"] == emb
    out.loc[m, "iccoh"] = E.loc[m, "iccoh_gene"]
    out.to_csv(R + dst, index=False)
    print("wrote", dst, out.shape)
