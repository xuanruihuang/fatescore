# fatescore

Score fate predictions against lineage-traced (clonal) fate, calibrated against CellRank's
connectivity kernel on the same graph and terminal states.

```python
from fatescore import score_table
table = score_table({"my_method": clone_scores}, kernel=kernel_clone_scores, labels=clone_labels,
                    kernel_settings={"n15_p30": ..., "n60_p50": ...})
```

Returns per method: clone-level AUROC; the kernel and its range over graph settings; the gain over
the kernel with a paired bootstrap over clones and the minimum detectable difference; the increment
over the kernel (cross-fitted logistic models: does the method carry fate information the walk
lacks?). Clone scores are cell scores averaged within clones (`clone_means`).

Self-check: `python tests/test_core.py` (run from this folder with the package on the path).
