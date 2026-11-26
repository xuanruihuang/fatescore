"""fatescore: score fate predictions against clonal fate, calibrated against a walk without direction.

A method's prediction counts as evidence of dynamics only where it beats CellRank's connectivity
kernel on the same graph and terminal states. This package returns the clone-level score, the gain
over the kernel with a paired bootstrap over clones, the minimum detectable difference, and the
increment over the kernel (information the walk lacks).
"""
from .core import clone_auroc, clone_means, gain_over_kernel, increment_over_kernel, mdd_factor, score_table

__all__ = ["clone_auroc", "clone_means", "gain_over_kernel", "increment_over_kernel", "mdd_factor",
           "score_table"]
__version__ = "0.1.0"
