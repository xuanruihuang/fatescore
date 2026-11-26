# fatescore

Code for *Does Agreement with Clonal Fate Measure Inferred Cell Dynamics?* (Xuanrui Huang, 2026).

Agreement between predicted and lineage-traced (clonal) fate is the test most often used to support
single-cell dynamics inferred by RNA velocity, optimal transport, flow matching or neural differential
equations. A random walk that ignores direction, on the same neighbour graph and towards the same
terminal states (CellRank's connectivity kernel), scores as well as every method we tested. This
repository contains

- `tool/fatescore`: the scoring tool. Given each method's fate predictions and a clone table, it
  returns the clone-level AUROC, the gain over the walk with a paired bootstrap interval over clones
  and the minimum detectable difference, and the increment over the walk (information the walk lacks).
- `code/`: every analysis script of the paper (data sets, method runs, scoring, simulation, figures).
- `reproduce.sh`: one stage per figure, table and in-text number; `REPRODUCE.md` maps each element of
  the paper to its stage and output file; `DATA.md` lists the public data sets.

## Install the tool

```
pip install .
python tool/tests/test_core.py
```

## Use it

```python
from fatescore import clone_means, score_table
# clone-level scores: cell scores averaged within clones
table = score_table({"my_method": method_clone_scores}, kernel=walk_clone_scores, labels=clone_labels,
                    kernel_settings={"n15_p30": walk_other_setting_scores})
```

## Reproduce the paper

`bash reproduce.sh <ROOT>`; see `REPRODUCE.md` for the environments, data and wall times. The
scoring and figure stages run in minutes from the method outputs; retraining every method takes days
on a GPU.

## Citation

Does Agreement with Clonal Fate Measure Inferred Cell Dynamics? Xuanrui Huang. arXiv, 2026.
