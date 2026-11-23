"""Render the V3.5 figure using adjacent aggregate CSV only."""
from __future__ import annotations
import csv
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
SEEDS = (20261107, 20261108, 20261109)
GROUPS = ('negative', 'middle', 'high')
COLORS = {1: '#465AB0', 5: '#C11155'}
STYLES = {1: '-', 5: (0, (4, 2))}
MARKERS = {1: 'o', 5: 'D'}

def plot():
    os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache'))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    with (HERE / 'FIG02_DATA.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 90

    def value(density, seed, group, metric, method):
        selected = [r for r in rows if (int(r['density_pct']), r['seed'], r['group'], r['metric'], r['method']) == (density, str(seed), group, metric, method)]
        assert len(selected) == 1, (density, seed, group, metric, method)
        return float(selected[0]['value'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.titleweight': 'semibold', 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'text.color': '#252A34', 'axes.labelcolor': '#252A34', 'xtick.color': '#454B56', 'ytick.color': '#454B56', 'axes.edgecolor': '#A0A6AF', 'axes.linewidth': 0.65, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'mathtext.fontset': 'dejavusans', 'path.simplify': False, 'svg.hashsalt': 'paper5-figure2-20261118'})
    (fig, (ax, bx)) = plt.subplots(1, 2, figsize=(7.16, 3.18))
    fig.subplots_adjust(left=0.083, right=0.985, bottom=0.225, top=0.795, wspace=0.38)
    for panel in (ax, bx):
        panel.spines[['top', 'right']].set_visible(False)
        panel.set_axisbelow(True)
        panel.grid(axis='y', color='#E5E7EB', linewidth=0.6)
        panel.tick_params(length=3, width=0.6)
    ax.set_title('(a) Learned gate behavior', loc='left', pad=10)
    bx.set_title('(b) Prediction-range MAE reduction', loc='left', pad=10)
    b = np.linspace(-1, 10, 1101)
    assert b[100] == 0
    x = np.log1p(np.maximum(b, 0))
    for density in (1, 5):
        curves = []
        for seed in SEEDS:
            a = value(density, seed, 'all', 'a', 'PCF')
            beta = value(density, seed, 'all', 'beta', 'PCF')
            w = 1 / (1 + np.exp(-(a - np.logaddexp(0, beta) * x)))
            assert np.all((w > 0) & (w < 1)) and np.all(np.diff(w) <= 1e-14)
            assert np.all(w[:101] == w[0])
            curves.append(w)
            ax.plot(b, w, color=COLORS[density], ls=STYLES[density], lw=0.8, alpha=0.3)
        ax.plot(b, np.mean(curves, axis=0), color=COLORS[density], ls=STYLES[density], lw=2.0)
    ax.axvline(0, color='#A0A6AF', ls=(0, (2, 3)), lw=0.65, zorder=0)
    ax.annotate('Constant for $b<0$', xy=(-0.55, 0.93), xytext=(3.6, 0.85), fontsize=7.5, color='#545B66', arrowprops=dict(arrowstyle='-', color='#9299A3', lw=0.65))
    ax.set(xlim=(-1, 10), ylim=(0, 1), xlabel='B3 prediction (s)', ylabel='NIMF fusion weight')
    ax.set_xticks([-1, 0, 2, 4, 6, 8, 10])
    ax.set_yticks(np.linspace(0, 1, 6))
    bx.axhline(0, color='#7E8590', lw=0.8, zorder=2)
    for (density, offset) in ((1, -0.18), (5, 0.18)):
        for (index, group) in enumerate(GROUPS):
            vals = np.array([value(density, seed, group, 'relative_mae_reduction', 'PCF_vs_fixed50') for seed in SEEDS])
            for (seed, relative) in zip(SEEDS, vals):
                fixed = value(density, seed, group, 'mae', 'fixed50')
                pcf = value(density, seed, group, 'mae', 'PCF')
                assert relative == 100 * (fixed - pcf) / fixed
            mean = value(density, 'mean', group, 'relative_mae_reduction', 'PCF_vs_fixed50')
            assert np.isclose(mean, vals.mean(), rtol=0, atol=1e-13)
            bx.scatter(index + offset + np.array([-0.12, -0.06, 0]), vals, s=15, marker=MARKERS[density], facecolors='none', edgecolors=COLORS[density], alpha=0.68, linewidths=0.85, zorder=3)
            bx.scatter(index + offset + 0.095, mean, s=50, marker=MARKERS[density], color=COLORS[density], edgecolor='white', linewidth=0.7, zorder=4)
    bx.set(xlim=(-0.5, 2.5), ylim=(-6, 38), ylabel='Relative MAE reduction (%)', xlabel='B3 prediction range (s)')
    bx.set_xticks([0, 1, 2], ['Negative\n$b<0$', 'Middle\n$0\\leq b<5$', 'High\n$b\\geq5$'])
    bx.set_yticks([-5, 0, 10, 20, 30])
    bx.text(0.98, 0.96, 'Positive favors PCF', transform=bx.transAxes, ha='right', va='top', fontsize=7.5, color='#545B66')
    handles = [Line2D([], [], color=COLORS[d], ls=STYLES[d], lw=1.7, marker=MARKERS[d], markersize=4, label=f'RT {d}%') for d in (1, 5)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 0.998), ncol=2, frameon=False, fontsize=8.5, handlelength=2.2, columnspacing=2.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels(), *bx.get_xticklabels(), *bx.get_yticklabels(), ax.xaxis.label, ax.yaxis.label, bx.xaxis.label, bx.yaxis.label, ax.title, bx.title, *fig.texts]:
        bb = label.get_window_extent(renderer)
        assert bb.x0 >= 0 and bb.y0 >= 0 and (bb.x1 <= fig.bbox.width) and (bb.y1 <= fig.bbox.height), label.get_text()
    stem = HERE / 'fig02_prediction_range_analysis'
    fig.savefig(stem.with_suffix('.pdf'), metadata={'Title': 'Prediction-conditioned fusion analysis', 'Author': 'Paper5'})
    fig.savefig(stem.with_suffix('.svg'), metadata={'Title': 'Prediction-conditioned fusion analysis'})
    fig.savefig(stem.with_suffix('.png'), dpi=600)
    fig.savefig(HERE / 'fig02_preview_7p16in.png', dpi=150)
    plt.close(fig)
    print('Saved single candidate: 7.16 x 3.18 in; vector PDF/SVG, 600-dpi PNG, 150-dpi preview.')
    print('Checks passed: 90 CSV rows, 6 frozen gates, 18 seed percentages, 6 arithmetic means, text bounds.')

if __name__ == '__main__':
    plot()
