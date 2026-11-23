"""Recreate the published table from public scalar results, without evaluation."""
import argparse
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = [('NIMF', 'NIMF95'), ('B3', 'B3-49'), ('fixed_half', 'Fixed 50/50'),
          ('constant_sigmoid', 'Constant gate'), ('B3_nonlinear', 'B3 nonlinear'),
          ('DHEM_PCW', 'DHEM-style PCW'), ('conditional', 'PCF'), ('RAQoS_independent', 'RA-QoS (independent)')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'build/table')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'results/final_per_seed.csv').open(newline='', encoding='utf-8') as f:
        per_seed = list(csv.DictReader(f))
    with (ROOT / 'results/final_aggregate.csv').open(newline='', encoding='utf-8') as f:
        aggregate = list(csv.DictReader(f))
    assert len(per_seed) == 48 and len(aggregate) == 16
    rows = []
    for method, label in LABELS:
        values = []
        for density in (1, 5):
            group = [r for r in per_seed if int(r['density_percent']) == density and r['method'] == method]
            assert {int(r['seed']) for r in group} == {20261107, 20261108, 20261109}
            saved = next(r for r in aggregate if int(r['density_percent']) == density and r['method'] == method)
            for metric in ('mae', 'rmse'):
                numbers = np.array([float(r[metric]) for r in group])
                mean, sd = float(numbers.mean()), float(numbers.std(ddof=1))
                np.testing.assert_allclose([mean, sd], [float(saved[metric+'_mean']), float(saved[metric+'_sample_sd'])], rtol=0, atol=1e-14)
                values.append(f'{mean:.4f} ± {sd:.4f}')
        rows.append((label, values))
    header = '| Method | 1% MAE | 1% RMSE | 5% MAE | 5% RMSE |\n|---|---:|---:|---:|---:|\n'
    md = header + '\n'.join('| ' + ' | '.join([label, *values]) + ' |' for label, values in rows)
    (args.out / 'table.md').write_text(md + '\n', encoding='utf-8')
    body = []
    for label, values in rows:
        cells = [label.replace('%', r'\%'), *(v.replace(' ± ', r' $\pm$ ') for v in values)]
        body.append(' & '.join(cells) + r' \\')
    tex = '\\begin{tabular}{lrrrr}\n\\hline\nMethod & 1\\% MAE & 1\\% RMSE & 5\\% MAE & 5\\% RMSE \\\\\n\\hline\n'
    tex += '\n'.join(body) + '\n\\hline\n\\end{tabular}\n'
    (args.out / 'table.tex').write_text(tex, encoding='utf-8')
    print('Verified 48 seed records and 64 aggregate numbers; saved Markdown/LaTeX table.')


if __name__ == '__main__':
    main()
