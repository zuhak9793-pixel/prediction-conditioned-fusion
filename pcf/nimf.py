"""Frozen NIMF95, native seconds and positive-correlation top-10 graph."""
import numpy as np
from NIMF import Model

def initialize(shape, dim, seed):
    rng = np.random.RandomState(seed)
    return (rng.rand(shape[0], dim), rng.rand(shape[1], dim))


def components(model, rows, cols, train_user_counts, train_service_counts, alpha):
    (p, q) = model.factors()
    w = model.weights()
    self_prediction = np.einsum('ij,ij->i', p[rows], q[cols])
    neighbor_prediction = np.einsum('ij,ij->i', (w @ p)[rows], q[cols])
    combined = model.predict(rows, cols)
    independent = alpha * self_prediction + (1 - alpha) * neighbor_prediction
    np.testing.assert_allclose(combined, independent, rtol=1e-12, atol=1e-12)
    support = (train_user_counts[rows] > 0) & (train_service_counts[cols] > 0) & (w.sum(1)[rows] > 0)
    assert np.isfinite(np.r_[self_prediction, neighbor_prediction, combined]).all()
    return (self_prediction, neighbor_prediction, combined, support, w, p, q)


def fit_fold(train, omitted_coords, shape, cfg, guard, on_step):
    (tu, ts, ty) = train
    (rows, cols) = omitted_coords
    ids = tu * shape[1] + ts
    testids = rows * shape[1] + cols
    assert len(np.unique(ids)) == len(ids) and (not np.intersect1d(ids, testids).size)
    matrix = np.zeros(shape)
    matrix[tu, ts] = ty
    assert np.count_nonzero(matrix) == len(ty) and (ty > 0).all()
    (p, q) = initialize(shape, cfg['dimension'], cfg['seed'])
    native = Model(matrix, p, q, cfg['lambda'], cfg['alpha'], cfg['etaInit'])
    for iteration in range(1, cfg['iterations'] + 1):
        guard()
        before = native.objective()
        step = native.step()
        after = native.objective()
        guard()
        assert np.isfinite([before, step, after]).all()
        on_step({'iteration': iteration, 'objective_before': before, 'objective': after, 'step': step, 'objective_increased': bool(after > before), 'unchecked_line_search_step': bool(step == cfg['etaInit'] / 2 ** 20)})
    uc = np.bincount(tu, minlength=shape[0])
    sc = np.bincount(ts, minlength=shape[1])
    values = components(native, rows, cols, uc, sc, cfg['alpha'])
    return (values, uc, sc)
