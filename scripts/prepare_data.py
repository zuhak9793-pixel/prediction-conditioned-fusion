"""Prepare a newly obtained response-time matrix using the frozen split recipe.

Never run against the original research project's sealed data or partitions.
This script is provided for independent users; release QA uses synthetic data only.
"""
import argparse
from pathlib import Path
import json
import numpy as np


def partitions(matrix):
    flat = matrix.ravel()
    eligible = np.flatnonzero(np.isfinite(flat) & (flat > 0))
    n = len(eligible)
    nf, nv = int(.2 * n), int(.05 * n)
    perm = np.random.default_rng(20261107).permutation(eligible)
    final, rest = perm[:nf], perm[nf:]
    roles = {}
    for seed in (20261107, 20261108, 20261109):
        seq = np.random.default_rng(seed).permutation(rest)
        val, pool = seq[:nv], seq[2 * nv:]
        roles[seed] = dict(validation=val, train01=pool[:int(.01 * n)],
                           train05=pool[:int(.05 * n)])
    return final, roles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', required=True, type=Path, help='Extracted dataset1/rtMatrix.txt')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists; preparation never overwrites a sealed partition')
    if 'private_data' in args.matrix.resolve().parts:
        parser.error('Do not use the original project private_data directory')
    matrix = np.loadtxt(args.matrix)
    if matrix.shape != (339, 5825):
        raise ValueError('Expected WS-DREAM 339 x 5825 response-time matrix')
    if np.count_nonzero(np.isfinite(matrix) & (matrix > 0)) != 1873838:
        raise ValueError('Eligible-observation count differs from the frozen data identity')
    final, roles = partitions(matrix)
    args.out.mkdir(parents=True)

    def save(name, indices, role, seed):
        u, s = indices // matrix.shape[1], indices % matrix.shape[1]
        np.savez_compressed(args.out / (name + '.npz'), role=role, seed=seed,
                            shape=matrix.shape, u=u, s=s, y=matrix.ravel()[indices])
        np.savez_compressed(args.out / (name + '_coords.npz'), u=u, s=s)

    save('reserved_test', final, 'reserved-test', -1)
    for seed, item in roles.items():
        save(f'{seed}_validation', item['validation'], 'development-validation', seed)
        for density in (1, 5):
            save(f'{seed}_train{density:02d}', item[f'train{density:02d}'], 'training', seed)
    (args.out / 'preparation.json').write_text(json.dumps(dict(shape=matrix.shape,
        eligible=1873838, final_count=len(final), validation_count=93691,
        training_counts={'1': 18738, '5': 93691},
        unused_role='The historical development-evaluation segment is skipped and never exported.',
        scope='Independent preparation; does not certify equality to private research partitions.'), indent=2), encoding='utf-8')
    print('Prepared independent role-pure partitions. Keep all generated files private.')


if __name__ == '__main__':
    main()
