"""Adapted DHEM-style K=35 confidence weighting, not full DHEM."""
from dataclasses import dataclass
import numpy as np
K = 35
USERS = 339

@dataclass
class Histogram:
    count: np.ndarray
    low: np.ndarray
    high: np.ndarray
    mass: np.ndarray


def bin_index(values, low, high):
    return np.minimum(K - 1, np.floor(K * (values - low) / (high - low)).astype(np.int64))


def histogram(u, y, users=USERS):
    assert u.shape == y.shape and np.isfinite(y).all()
    assert np.issubdtype(u.dtype, np.integer) and ((u >= 0) & (u < users)).all()
    count = np.bincount(u, minlength=users)
    (low, high) = (np.full(users, np.inf), np.full(users, -np.inf))
    np.minimum.at(low, u, y)
    np.maximum.at(high, u, y)
    mass = np.zeros((users, K))
    nonconstant = high[u] > low[u]
    index = np.zeros(len(y), dtype=np.int64)
    index[nonconstant] = bin_index(y[nonconstant], low[u[nonconstant]], high[u[nonconstant]])
    np.add.at(mass, (u, index), 1)
    np.testing.assert_array_equal(mass.sum(axis=1), count)
    mass /= np.maximum(count[:, None], 1)
    np.testing.assert_allclose(mass.sum(axis=1), count > 0, rtol=0, atol=5e-16)
    return Histogram(count, low, high, mass)


def confidence(h, u, prediction):
    assert np.isfinite(prediction).all() and prediction.shape == u.shape
    empty = h.count[u] == 0
    outside = ~empty & ((prediction < h.low[u]) | (prediction > h.high[u]))
    active = ~empty & ~outside
    constant = active & (h.low[u] == h.high[u])
    regular = active & ~constant
    conf = np.zeros(len(u))
    index = bin_index(prediction[regular], h.low[u[regular]], h.high[u[regular]])
    conf[regular] = h.mass[u[regular], index]
    conf[constant] = 1.0
    return (conf, {'outside_training_range': int(outside.sum()), 'empty_user': int(empty.sum()), 'in_range_empty_bin': int((regular & (conf == 0)).sum()), 'constant_point_match': int(constant.sum()), 'zero_confidence': int((conf == 0).sum())})


def fuse(h, u, nimf, b3):
    (cn, dn) = confidence(h, u, nimf)
    (cb, db) = confidence(h, u, b3)
    total = cn + cb
    positive = total > 0
    weights = np.full(len(u), 0.5)
    weights[positive] = cn[positive] / total[positive]
    pred = 0.5 * nimf + 0.5 * b3
    pred[positive] = (cn[positive] * nimf[positive] + cb[positive] * b3[positive]) / total[positive]
    assert np.isfinite(pred).all() and ((weights >= 0) & (weights <= 1)).all()
    assert (pred >= np.minimum(nimf, b3) - 1e-12).all()
    assert (pred <= np.maximum(nimf, b3) + 1e-12).all()
    detail = {'n': len(u), 'histogram_observations': int(h.count.sum()), 'histogram_empty_users': int((h.count == 0).sum()), 'histogram_constant_nonempty_users': int(((h.count > 0) & (h.low == h.high)).sum()), 'NIMF': dn, 'B3': db, 'zero_total_half_fallback': int((~positive).sum()), 'NIMF_only': int(((cn > 0) & (cb == 0)).sum()), 'B3_only': int(((cb > 0) & (cn == 0)).sum()), 'both_positive': int(((cn > 0) & (cb > 0)).sum()), 'mean_NIMF_weight_including_fallback': float(weights.mean())}
    assert sum((detail[k] for k in ['zero_total_half_fallback', 'NIMF_only', 'B3_only', 'both_positive'])) == len(u)
    return (pred, detail)
