"""Separate training, inference and explicitly requested evaluation stages."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
from pathlib import Path
import numpy as np
from .data import load_triples, metrics, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('oof', 'train'):
        p = sub.add_parser(name)
        p.add_argument('--train', required=True, type=Path)
        p.add_argument('--seed', required=True, type=int, choices=[20261107, 20261108, 20261109])
        p.add_argument('--out', required=True, type=Path)
    p = sub.add_parser('fit-gate')
    p.add_argument('--oof', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--development-ablation', action='store_true',
                   help='Fit the unrestricted-slope development control as well')
    p = sub.add_parser('predict')
    p.add_argument('--train', required=True, type=Path)
    p.add_argument('--models', required=True, type=Path)
    p.add_argument('--gate', required=True, type=Path)
    p.add_argument('--coords', required=True, type=Path, help='Only u/s members are read')
    p.add_argument('--out', required=True, type=Path)
    p = sub.add_parser('score')
    p.add_argument('--targets', required=True, type=Path)
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--role', required=True, choices=['development-validation', 'reserved-test'])
    p.add_argument('--acknowledge-reserved-test', action='store_true')
    p.add_argument('--out', required=True, type=Path)
    p = sub.add_parser('train-raqos')
    p.add_argument('--train', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p = sub.add_parser('predict-raqos')
    p.add_argument('--train', required=True, type=Path)
    p.add_argument('--models', required=True, type=Path)
    p.add_argument('--coords', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists; choose a new output to preserve prior evidence')
    if args.command in ('oof', 'predict', 'predict-raqos') and args.out.suffix != '.npz':
        parser.error('Array output must have an explicit .npz suffix')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = read_json(ROOT / 'configs/frozen.json')
    if args.command in ('oof', 'train'):
        from . import pipeline
        train, shape = load_triples(args.train, 'training')
        with np.load(args.train, allow_pickle=False) as z:
            if int(z['seed']) != args.seed:
                parser.error('Training seed does not match requested seed')
        if args.command == 'oof':
            pipeline.make_oof(train, shape, config, args.seed, args.out)
        else:
            args.out.mkdir()
            pipeline.train_full(train, shape, config, args.seed, args.out)
    elif args.command == 'fit-gate':
        from .pipeline import fit_readouts
        fit_readouts(args.oof, config, args.out, args.development_ablation)
    elif args.command == 'predict':
        from .pipeline import predict_full
        train, shape = load_triples(args.train, 'training')
        with np.load(args.coords, allow_pickle=False) as z:
            rows, cols = z['u'], z['s']
        out = predict_full(train, shape, (rows, cols), args.models, args.gate)
        np.savez_compressed(args.out, u=rows, s=cols, **out)
    elif args.command == 'score':
        if args.role == 'reserved-test' and not args.acknowledge_reserved_test:
            parser.error('Reserved-test evaluation requires --acknowledge-reserved-test')
        train_keys = {'role', 'shape', 'seed', 'u', 's', 'y'}
        (u, s, y), _ = load_triples(args.targets, args.role)
        with np.load(args.predictions, allow_pickle=False) as z:
            np.testing.assert_array_equal(u, z['u'])
            np.testing.assert_array_equal(s, z['s'])
            result = {k: metrics(y, z[k]) for k in z.files if k not in train_keys}
            if not result:
                raise ValueError('No prediction methods')
        write_json(args.out, dict(role=args.role, metrics=result))
    elif args.command == 'train-raqos':
        from . import raqos_training
        train, shape = load_triples(args.train, 'training')
        cfg = read_json(ROOT / 'configs/raqos.json')
        if list(shape) != cfg['shape']:
            raise ValueError('Frozen RA-QoS requires the WS-DREAM shape')
        raqos_training.train(train, cfg, args.out)
    else:
        from . import raqos_training
        train, shape = load_triples(args.train, 'training')
        with np.load(args.coords, allow_pickle=False) as z:
            u, s = z['u'], z['s']
        prediction = raqos_training.infer(train, shape, args.models, u, s)
        np.savez_compressed(args.out, u=u, s=s, RAQoS_independent=prediction)
    print('Saved', args.out)


if __name__ == '__main__':
    main()
