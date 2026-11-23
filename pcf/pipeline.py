"""Training-only OOF construction, frozen full fits and coordinate inference."""
import numpy as np
from . import b3, fusion, pcw
from .data import folds, write_json, read_json, training_identity

STATE_KEYS = ('mu', 'scale', 'bu', 'bs', 'p', 'q')


def expert_predictions(train, coordinates, shape, config, seed):
    from . import nimf
    native_config = dict(config['nimf'], seed=seed)
    values, _, _ = nimf.fit_fold(train, coordinates, shape, native_config,
                                lambda: None, lambda row: None)
    state, b, _, _ = b3.fit_fold(train, coordinates, shape, config['B3'], seed,
                                config['B3_sweeps'])
    return values[2], b, values[-2:], state


def make_oof(train, shape, config, seed, output):
    u, s, y = train
    fold = folds(len(y), seed)
    n, b, confidence = (np.full(len(y), np.nan) for _ in range(3))
    filled = np.zeros(len(y), dtype=np.int8)
    for f in range(3):
        hold = fold == f
        keep = ~hold
        fit_data = (u[keep], s[keep], y[keep])
        n[hold], b[hold], _, _ = expert_predictions(fit_data, (u[hold], s[hold]),
                                                   shape, config, seed)
        hist = pcw.histogram(u[keep], y[keep], users=shape[0])
        confidence[hold] = pcw.fuse(hist, u[hold], n[hold], b[hold])[0]
        filled[hold] += 1
    if not (np.all(filled == 1) and np.isfinite(np.r_[n, b, confidence]).all()):
        raise RuntimeError('Incomplete or nonfinite OOF predictions')
    np.savez_compressed(output, role='training-oof', shape=shape, seed=seed,
                        training_identity=training_identity(train, shape),
                        u=u, s=s, y=y, folds=fold, NIMF=n, B3=b, DHEM_PCW=confidence)


def fit_readouts(oof_path, config, output, development_ablation=False):
    with np.load(oof_path, allow_pickle=False) as archive:
        if str(archive['role']) != 'training-oof':
            raise ValueError('Only training OOF data may fit a gate')
        n, b, y = (archive[k] for k in ('NIMF', 'B3', 'y'))
        seed = int(archive['seed'])
        identity = str(archive['training_identity'])
        if identity != training_identity((archive['u'], archive['s'], y), archive['shape']):
            raise ValueError('OOF training identity mismatch')
        np.testing.assert_array_equal(archive['folds'], folds(len(y), seed))
        if not all(x.shape == y.shape and np.isfinite(x).all() for x in (n, b, y)):
            raise ValueError('Malformed OOF data')
    kinds = ['PCF', 'constant_sigmoid', 'B3_nonlinear']
    if development_ablation:
        kinds.append('unconstrained_sigmoid')
    coefficients, solvers = {}, {}
    for kind in kinds:
        theta, record = fusion.fit(kind, n, b, y, config)
        coefficients[kind] = theta.tolist()
        solvers[kind] = record
    write_json(output, dict(seed=seed, role='training-oof-fitted',
                            training_identity=identity,
                            config=config, development_ablation=development_ablation,
                            coefficients=coefficients, solvers=solvers))


def train_full(train, shape, config, seed, output):
    empty = np.empty(0, dtype=np.int64)
    _, _, (p, q), state = expert_predictions(train, (empty, empty), shape, config, seed)
    np.savez_compressed(output / 'nimf.npz', U=p, S=q)
    np.savez_compressed(output / 'b3.npz', **dict(zip(STATE_KEYS, state)))
    write_json(output / 'training.json', dict(seed=seed, shape=shape, config=config,
               training_identity=training_identity(train, shape)))


def predict_full(train, shape, coordinates, model_dir, readout_path):
    from NIMF import Model
    u, s, y = train
    meta = read_json(model_dir / 'training.json')
    readout = read_json(readout_path)
    if meta['shape'] != list(shape) or meta['seed'] != readout['seed'] or meta['config'] != readout['config']:
        raise ValueError('Expert/readout configuration mismatch')
    if meta['training_identity'] != readout['training_identity'] or meta['training_identity'] != training_identity(train, shape):
        raise ValueError('Expert/readout training identity mismatch')
    cfg = meta['config']['nimf']
    matrix = np.zeros(shape)
    matrix[u, s] = y
    with np.load(model_dir / 'nimf.npz', allow_pickle=False) as archive:
        native = Model(matrix, archive['U'], archive['S'], cfg['lambda'], cfg['alpha'], cfg['etaInit'])
    with np.load(model_dir / 'b3.npz', allow_pickle=False) as archive:
        state = tuple(archive[k] for k in STATE_KEYS)
    rows, cols = coordinates
    n = native.predict(rows, cols)
    b = b3.predict(state, rows, cols)
    out = fusion.predictions(n, b, readout['coefficients'], readout['development_ablation'])
    out['DHEM_PCW'] = pcw.fuse(pcw.histogram(u, y, users=shape[0]), rows, n, b)[0]
    if not all(p.shape == rows.shape and np.isfinite(p).all() for p in out.values()):
        raise ValueError('Nonfinite or malformed prediction')
    return out
