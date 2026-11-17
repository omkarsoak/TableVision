"""Bootstrap confidence intervals, so that model comparisons on ~500 samples can be judged properly."""
from typing import Sequence, Tuple

import numpy as np


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """(mean, lower, upper) percentile-bootstrap CI of the mean."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Paired comparison of two models scored on the SAME samples (aligned by position).

    Returns (mean(a - b), lower, upper, p) where p is the bootstrap probability that the true difference is <= 0
    (small p: a really is better than b)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)), float((means <= 0).mean())
