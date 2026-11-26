"""Counts the draft quotes that no other script writes to a file.

Usage: 63_misc_counts.py <d2_labels.tsv> <mc_d2.csv> <out.csv>
  - day-2 cells and clones of the continuous task (at least 3 neutrophil or
    monocyte descendants; the clone's dominant outcome need not be either);
  - of the 212 clones of the matched two-well comparison (22_matched_ceiling.py),
    how many belong to the main task, and how many are outside it because they
    have only 2 mature descendants.
"""
import sys
import pandas as pd

T = pd.read_csv(sys.argv[1], sep="\t")
rows = []
cont = T[(T["n_neu"] + T["n_mono"]) >= 3]
rows.append(dict(quantity="continuous task, day-2 cells", value=len(cont)))
rows.append(dict(quantity="continuous task, clones", value=cont["clone"].nunique()))
main = T[(T["n_sis"] >= 3) & T["dom"].isin(["Neutrophil", "Monocyte"])]
mc_ok = T[T["dom_w1"].isin(["Neutrophil", "Monocyte"])
          & T["dom_w2"].isin(["Neutrophil", "Monocyte"])]
both = set(mc_ok["clone"])
rows.append(dict(quantity="two-well clones with a day-2 cell, both wells neu/mono-dominant",
                 value=len(both)))
rows.append(dict(quantity="  of which in the main task", value=len(both & set(main["clone"]))))
out = mc_ok[~mc_ok["clone"].isin(set(main["clone"]))].groupby("clone")["n_sis"].first()
rows.append(dict(quantity="  of which outside the main task", value=len(out)))
rows.append(dict(quantity="  outside, with exactly 2 mature descendants",
                 value=int((out == 2).sum())))
R = pd.DataFrame(rows)
R.to_csv(sys.argv[3], index=False)
print(R.to_string(index=False))
