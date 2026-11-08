"""Sampling and DKW statistics for Strategy B (no model/GPU dependencies)."""
import math

import numpy as np


def calibration_epochs(max_epoch, interval):
    if interval < 1:
        raise ValueError('Calibration interval must be positive')
    return list(range(interval, max_epoch, interval))


def adjusted_quantile(samples, calibrations, q=0.95, delta=0.05):
    if samples < 1 or calibrations < 1 or not 0 < q < 1 or not 0 < delta < 1:
        raise ValueError('Invalid DKW parameters')
    epsilon = math.sqrt(math.log(2 * calibrations / delta) / (2 * samples))
    level = q + epsilon
    if level > 1:
        raise ValueError('DKW-adjusted quantile exceeds 1; increase sample count')
    return epsilon, level, math.ceil(samples * level)


def select_budget(probabilities, calibrations, tau=0.90, q=0.95, delta=0.05):
    """Rows are descending top-50 probabilities, zero-padded for small graphs."""
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 50 or not len(p) or
            not np.isfinite(p).all() or (p < 0).any() or
            not np.allclose(p.sum(axis=1), 1, atol=1e-5) or
            (np.diff(p, axis=1) > 1e-6).any() or not 0 < tau <= 1):
        raise ValueError('Expected normalized, sorted top-50 probabilities')
    cumulative = np.cumsum(p / p.sum(axis=1, keepdims=True), axis=1)
    # Normalize and pin the final nonzero entry, including for tau=1.
    lengths = (p > 0).sum(axis=1)
    cumulative[np.arange(len(p)), lengths - 1] = 1.0
    cumulative = np.maximum.accumulate(cumulative, axis=1)
    required = np.minimum((cumulative < tau).sum(axis=1) + 1, 50)
    epsilon, level, rank = adjusted_quantile(len(p), calibrations, q, delta)
    k = int(np.partition(required, rank - 1)[rank - 1])
    mass = cumulative[:, k - 1]
    summary = dict(k=k, samples=len(p), planned_calibrations=calibrations,
                   dkw_epsilon=epsilon, quantile_level=level, quantile_rank=rank,
                   required_mean=float(required.mean()),
                   required_p50=float(np.quantile(required, 0.5)),
                   required_p95=float(np.quantile(required, 0.95)),
                   required_max=int(required.max()),
                   mass_mean=float(mass.mean()), mass_min=float(mass.min()),
                   mass_p05=float(np.quantile(mass, 0.05)),
                   empirical_coverage=float((required <= k).mean()),
                   coverage_lower_bound=max(0.0, float((required <= k).mean()) - epsilon),
                   required_histogram=np.bincount(required, minlength=51)[1:].tolist())
    return k, required, mass, summary


def compare_calibrations(probabilities, required, previous_p, previous_required):
    """Unpaired distribution diagnostics; not a temporal coverage guarantee."""
    p = np.asarray(probabilities, dtype=np.float64).mean(axis=0)
    old = np.asarray(previous_p, dtype=np.float64).mean(axis=0)
    middle = (p + old) / 2
    def kl(a):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / middle[mask])))
    cdf = np.cumsum(np.bincount(required, minlength=51)[1:]) / len(required)
    old_cdf = np.cumsum(np.bincount(previous_required, minlength=51)[1:]) / len(previous_required)
    return dict(mean_profile_js_divergence=(kl(p) + kl(old)) / 2,
                required_budget_cdf_distance=float(np.max(np.abs(cdf - old_cdf))),
                required_budget_wasserstein=float(np.abs(cdf - old_cdf).sum()))
