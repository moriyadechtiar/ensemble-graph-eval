"""Disagreement between judges, and risk.

``disagreement`` turns the scores several judges gave one prediction into
uncertainty statistics. A prediction on which the judges disagree is more
likely to be wrong, so these statistics can rank predictions for review.
"""
from itertools import combinations
from typing import Dict, Iterable, Optional

import numpy as np


def disagreement(scores: Iterable[Optional[float]]) -> Dict[str, float]:
    """Uncertainty statistics for one prediction, from judge scores in [0, 1].

    cv        coefficient of variation (standard deviation divided by the mean)
    std, variance, range, iqr
    pairwise  mean absolute difference between pairs of judges
    entropy   Shannon entropy of the normalised scores
    margin    1 minus the gap between the two highest scores
    """
    s = np.array([x for x in scores if x is not None and np.isfinite(x)], dtype=float)
    names = ("cv", "entropy", "range", "pairwise", "margin", "iqr", "variance", "std")
    if s.size == 0:
        return {**{k: float("nan") for k in names}, "n_judges": 0}
    mean = s.mean()
    p = s / (s.sum() + 1e-8)
    if p.sum() > 0:
        q = p / p.sum()
        entropy = float(-(q[q > 0] * np.log(q[q > 0])).sum())
    else:
        entropy = 0.0
    top = np.sort(s)[::-1]
    return {
        "cv": float(s.std() / (mean + 1e-8)) if mean > 1e-8 else 0.0,
        "entropy": entropy,
        "range": float(s.max() - s.min()) if s.size > 1 else 0.0,
        "pairwise": float(np.mean([abs(a - b) for a, b in combinations(s, 2)])) if s.size >= 2 else 0.0,
        "margin": float(1 - (top[0] - top[1])) if s.size >= 2 else 0.0,
        "iqr": float(np.percentile(s, 75) - np.percentile(s, 25)) if s.size >= 4 else 0.0,
        "variance": float(s.var()),
        "std": float(s.std()),
        "n_judges": int(s.size),
    }


def mean_risk(scores: Iterable[Optional[float]]) -> float:
    """Mean risk over judges, where each judge's risk is 1 minus its score."""
    s = np.array([x for x in scores if x is not None and np.isfinite(x)], dtype=float)
    return float((1.0 - s).mean()) if s.size else float("nan")


def select_for_review(risks: Iterable[float], budget: float = 0.10):
    """Indices of the highest-risk predictions, for a review budget given as a share of all predictions."""
    r = np.asarray(list(risks), dtype=float)
    k = max(1, int(round(budget * r.size)))
    return [int(i) for i in np.argsort(-np.nan_to_num(r, nan=-np.inf), kind="stable")[:k]]


def calibrate(risks: Iterable[float], is_error: Iterable[int], n_splits: int = 5, random_state: int = 42):
    """Out-of-sample isotonic calibration of a risk score into a probability of error.

    Needs scikit-learn. Returns one probability per prediction.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import KFold
    x, y = np.asarray(list(risks), dtype=float), np.asarray(list(is_error), dtype=int)
    out = np.zeros_like(x, dtype=float)
    for train, test in KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(x):
        iso = IsotonicRegression(out_of_bounds="clip").fit(x[train], y[train])
        out[test] = iso.transform(x[test])
    return out
