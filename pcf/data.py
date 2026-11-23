"""Role-pure input contracts; fitting cannot consume evaluation archives."""
from pathlib import Path
import hashlib
import json
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_triples(path, required_role):
    with np.load(path, allow_pickle=False) as archive:
        role = str(archive['role'])
        if role != required_role:
            raise ValueError(f'Expected {required_role}, received {role}')
        shape = tuple(int(x) for x in archive['shape'])
        u, s, y = (archive[k] for k in ('u', 's', 'y'))
    validate(u, s, y, shape)
    return (u, s, y), shape


def validate(u, s, y, shape):
    if (len(shape) != 2 or min(shape) <= 0 or u.ndim != 1 or s.shape != u.shape
            or y.shape != u.shape or len(y) == 0):
        raise ValueError('Nonempty aligned triples and positive matrix shape required')
    if u.dtype.kind not in 'iu' or s.dtype.kind not in 'iu':
        raise ValueError('Integer coordinates required')
    if not ((u >= 0) & (u < shape[0]) & (s >= 0) & (s < shape[1])).all():
        raise ValueError('Coordinates outside shape')
    if not np.isfinite(y).all() or not (y > 0).all():
        raise ValueError('Finite positive targets in native seconds required')
    if len(np.unique(u * shape[1] + s)) != len(y):
        raise ValueError('Duplicate coordinates')


def folds(size, seed):
    order = np.random.RandomState(seed).permutation(size)
    result = np.empty(size, dtype=np.int8)
    result[order] = np.arange(size) % 3
    return result


def training_identity(train, shape):
    """Bind generated states/readouts to the exact ordered training observations."""
    digest = hashlib.sha256(np.asarray(shape, dtype='<i8').tobytes())
    for values, dtype in zip(train, ('<i8', '<i8', '<f8')):
        digest.update(np.asarray(values, dtype=dtype).tobytes())
    return digest.hexdigest()


def metrics(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 1 or predictions.shape != targets.shape or not len(targets):
        raise ValueError('Nonempty aligned one-dimensional arrays required')
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError('Nonfinite values')
    error = predictions - targets
    return dict(n=len(error), mae=float(np.mean(np.abs(error))),
                rmse=float(np.sqrt(np.mean(error * error))))
