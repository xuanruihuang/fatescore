"""Self-checks on synthetic clones: a copy of the kernel gains nothing and adds nothing; an
independent informative signal adds information; the MDD factor equals 0.714."""
import numpy as np
from fatescore import gain_over_kernel, increment_over_kernel, mdd_factor


def make(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    a, b = rng.normal(size=n), rng.normal(size=n)
    k = 1 / (1 + np.exp(-(0.6 * (2 * y - 1) + a)))          # kernel: state signal
    m = 1 / (1 + np.exp(-(0.8 * (2 * y - 1) + b)))          # independent signal
    return y, k, m


def test_mdd():
    assert abs(mdd_factor() - 0.7147) < 1e-3


def test_copy_of_kernel():
    y, k, _ = make()
    g = gain_over_kernel(k, k, y, n_boot=200)
    assert g["gain"] == 0 and g["verdict"] == "not separated"
    inc = increment_over_kernel(k, k, y, reps=3, n_boot=200)
    assert abs(inc["increment"]) < 0.01 and not inc["adds"]


def test_independent_signal_adds():
    y, k, m = make()
    inc = increment_over_kernel(m, k, y, reps=3, n_boot=200)
    assert inc["adds"] and inc["increment"] > 0.05


if __name__ == "__main__":
    test_mdd(); test_copy_of_kernel(); test_independent_signal_adds(); print("all tests passed")
