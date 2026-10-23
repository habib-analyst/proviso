"""Metrics and paired-bootstrap statistics."""

import numpy as np

from .traces import UNSAFE


def _label(ep):
    return 1 if ep.label == UNSAFE else 0


def decisions_to_preds(episodes, decisions):
    return [1 if d else 0 for d in decisions]


def f1_score_binary(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def f1_of(episodes, decide_fn):
    y_true = [_label(ep) for ep in episodes]
    y_pred = [1 if decide_fn(ep) else 0 for ep in episodes]
    return f1_score_binary(y_true, y_pred)


def paired_bootstrap_f1_gap(episodes, decide_a, decide_b, n_boot=2000,
                            seed=7):
    """Paired bootstrap over episodes for F1(a) - F1(b).

    Same resampled episode set is scored under both decision functions, so
    the gap's CI is paired. Returns (gap, lo, hi) with 95% percentile CI.
    """
    rng = np.random.RandomState(seed)
    n = len(episodes)
    ya = np.array([1 if decide_a(ep) else 0 for ep in episodes], dtype=int)
    yb = np.array([1 if decide_b(ep) else 0 for ep in episodes], dtype=int)
    yt = np.array([_label(ep) for ep in episodes], dtype=int)
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        gaps[b] = f1_score_binary(yt[idx], ya[idx]) - f1_score_binary(yt[idx], yb[idx])
    gap = f1_score_binary(yt, ya) - f1_score_binary(yt, yb)
    return gap, float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def bootstrap_mean_gap(pairs_a_correct, pairs_b_correct, n_boot=2000,
                       seed=7):
    """Paired bootstrap on per-episode correctness differences."""
    rng = np.random.RandomState(seed)
    a = np.asarray(pairs_a_correct, dtype=float)
    b = np.asarray(pairs_b_correct, dtype=float)
    diffs = a - b
    n = len(diffs)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        boots[i] = diffs[idx].mean()
    return float(diffs.mean()), float(np.percentile(boots, 2.5)), float(
        np.percentile(boots, 97.5))


def brier_score(y_true, p):
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y_true) ** 2))
