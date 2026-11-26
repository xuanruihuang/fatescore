"""What the CellTag-multi deposit could supply for this question.

Usage: 25_celltagmulti_clones.py <hsc_clone_table.csv> <sra_runinfo.csv> <out.csv>

The paper reports 2,227 haematopoiesis clones spanning its state and fate
samples. That count pools two assays run on separate aliquots of cells, so it
is not the number of clones whose fate is measured and whose transcriptome is
profiled at the state sample. This recounts the authors' own clone table under
the definition this question needs, and summarises what the raw sequence
archive holds, since the splicing layer would have to be quantified from it.
"""
import sys

import pandas as pd

clone_table, runinfo, out_path = sys.argv[1:4]

cl = pd.read_csv(clone_table).groupby("clone.id").first()
rna2, rna5 = "# of D2 cells (RNA)", "# of D5 cells (RNA)"
at2, at5 = "# of D2 cells (ATAC)", "# of D5 cells (ATAC)"
both2, both5 = "# of D2 cells (RNA & ATAC)", "# of D5 cells (RNA & ATAC)"

rows = [
    ("clones in the table", len(cl)),
    ("clones spanning state and fate, either assay",
     int(((cl[both2] >= 1) & (cl[both5] >= 1)).sum())),
    ("of those, with a transcriptome in the state sample",
     int(((cl[rna2] >= 1) & (cl[both5] >= 1)).sum())),
    ("with transcriptomes on both sides",
     int(((cl[rna2] >= 1) & (cl[rna5] >= 1)).sum())),
    ("with ATAC on both sides",
     int(((cl[at2] >= 1) & (cl[at5] >= 1)).sum())),
    ("with both assays on both sides",
     int(((cl[rna2] >= 1) & (cl[rna5] >= 1)
          & (cl[at2] >= 1) & (cl[at5] >= 1)).sum())),
]

R = pd.read_csv(runinfo)
rows.append(("sra runs", int(len(R))))
rows.append(("sra total gb", round(float(R["size_MB"].sum()) / 1024.0, 1)))
for s, n in R["LibraryStrategy"].value_counts().items():
    rows.append(("sra runs, %s" % s, int(n)))

pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(out_path, index=False)
print(pd.DataFrame(rows, columns=["quantity", "value"]).to_string(index=False))
