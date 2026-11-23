"""Frozen rank-8 B3 updates; training-only population standardization."""
import numpy as np
from scipy.sparse import csr_matrix

def block(opposite, adjacency, residual, lam, extra=0.0, tau=1.0):
    """All entities in a block use one frozen opposite-factor array."""
    rank = opposite.shape[1]
    gram = (adjacency @ (opposite[:, :, None] * opposite[:, None, :]).reshape(len(opposite), -1)).reshape(-1, rank, rank)
    h = residual @ opposite
    eye = np.eye(rank)
    penalty = np.asarray(lam)[:, None, None] * eye
    if extra:
        inverse_design = np.linalg.solve(gram + tau * eye, np.broadcast_to(eye, gram.shape))
        penalty = penalty + extra * tau * inverse_design
    return (np.linalg.solve(gram + penalty, h[..., None])[..., 0], gram, penalty)


def predict(state, u, s):
    (mu, scale, bu, bs, p, q) = state
    return mu + scale * (bu[u] + bs[s] + np.einsum('ij,ij->i', p[u], q[s]))


def metrics(y, pred):
    e = pred - y
    return {'mae': float(np.mean(abs(e))), 'rmse': float(np.sqrt(np.mean(e * e)))}


def fit_fold(train, omitted_coords, shape, config, seed, sweeps, guard=lambda : None, on_sweep=lambda r: None):
    (u, s, y) = train
    (ou, os) = omitted_coords
    (nu, ns) = shape
    assert len(np.unique(u * ns + s)) == len(y) and (not np.intersect1d(u * ns + s, ou * ns + os).size)
    assert np.isfinite(y).all() and (y > 0).all() and (config['method'] == 'B3')
    (mu, scale) = (float(y.mean()), max(float(y.std()), 1e-12))
    z = (y - mu) / scale
    (cu, cs) = (np.bincount(u, minlength=nu), np.bincount(s, minlength=ns))
    rng = np.random.default_rng(seed)
    (p, q) = (rng.normal(0, 0.1, (nu, 8)), rng.normal(0, 0.1, (ns, 8)))
    p[cu == 0] = 0
    q[cs == 0] = 0
    (bu, bs) = (np.zeros(nu), np.zeros(ns))
    lu = np.full(nu, config['lambda'])
    ls = np.full(ns, config['lambda'])
    lb = config['bias_lambda']
    adj = csr_matrix((np.ones(len(y)), (u, s)), shape=shape)
    for sweep in range(1, sweeps + 1):
        guard()
        interaction = np.einsum('ij,ij->i', p[u], q[s])
        bu = np.bincount(u, weights=z - bs[s] - interaction, minlength=nu) / (cu + lb)
        bs = np.bincount(s, weights=z - bu[u] - interaction, minlength=ns) / (cs + lb)
        residual = csr_matrix((z - bu[u] - bs[s], (u, s)), shape=shape)
        (p, _, _) = block(q, adj, residual, lu, config['lambda1'], config['tau'])
        (q, _, _) = block(p, adj.T, residual.T, ls, config['lambda1'], config['tau'])
        state = (mu, scale, bu, bs, p, q)
        assert all((np.isfinite(a).all() for a in state))
        assert np.count_nonzero(p[cu == 0]) == np.count_nonzero(q[cs == 0]) == np.count_nonzero(bu[cu == 0]) == np.count_nonzero(bs[cs == 0]) == 0
        on_sweep(dict(sweep=sweep, training=metrics(y, predict(state, u, s)), P_norm=float(np.linalg.norm(p)), Q_norm=float(np.linalg.norm(q))))
    return (state, predict(state, ou, os), cu, cs)
