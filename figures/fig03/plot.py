"""Render the V3.5 figure using adjacent aggregate CSV only."""
from __future__ import annotations
import csv
import math
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
SEEDS = {20261107, 20261108, 20261109}
METHODS = {'conditional': ('PCF', '#C11155', 'D', 49), 'B3': ('B3-49', '#465AB0', 's', 33), 'RAQoS_independent': ('RA-QoS', '#5BA7D9', 'o', 36), 'fixed_half': ('Fixed 50/50', '#9CA3AF', '^', 34), 'constant_sigmoid': ('Constant gate', '#CBD5E1', 'o', 48), 'B3_nonlinear': ('B3 nonlinear calibration', '#697586', 'v', 32)}
ALL_METHODS = set(METHODS) | {'NIMF', 'DHEM_PCW'}
EXPECTED_PARETO = {1: {'RAQoS_independent', 'B3_nonlinear', 'conditional', 'B3'}, 5: {'RAQoS_independent', 'conditional', 'B3'}}

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def pareto(rows):
    """Minimize both exact saved arithmetic means; no rounding/tolerance."""
    return {r['method'] for r in rows if not any((float(s['mae_mean_s']) <= float(r['mae_mean_s']) and float(s['rmse_mean_s']) <= float(r['rmse_mean_s']) and (float(s['mae_mean_s']) < float(r['mae_mean_s']) or float(s['rmse_mean_s']) < float(r['rmse_mean_s'])) for s in rows))}

def load_data():
    rows = read_csv(HERE / 'FIG03_DATA.csv')
    assert len(rows) == 16
    assert {(int(r['density_percent']), r['method']) for r in rows} == {(d, m) for d in (1, 5) for m in ALL_METHODS}
    for density in (1, 5):
        group = [r for r in rows if int(r['density_percent']) == density]
        assert pareto(group) == EXPECTED_PARETO[density]
        for row in group:
            assert int(row['seeds']) == 3
            assert int(row['displayed']) == int(row['method'] in METHODS)
            assert int(row['pareto_efficient_all_eight']) == int(row['method'] in pareto(group))
    return rows

def plot():
    os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.transforms import Bbox
    rows = load_data()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.edgecolor': '#9CA3AF', 'axes.linewidth': 0.65, 'text.color': '#27303D', 'axes.labelcolor': '#27303D', 'xtick.color': '#545E6C', 'ytick.color': '#545E6C', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    (fig, axes) = plt.subplots(1, 2, figsize=(7.16, 3.58))
    fig.subplots_adjust(left=0.087, right=0.98, bottom=0.297, top=0.9, wspace=0.285)
    configs = {1: ((0.658, 0.819), (1.594, 1.738), [0.66, 0.69, 0.72, 0.75, 0.78, 0.81], [1.6, 1.64, 1.68, 1.72]), 5: ((0.488, 0.58), (1.316, 1.406), [0.49, 0.51, 0.53, 0.55, 0.57], [1.32, 1.34, 1.36, 1.38, 1.4])}
    labels = {1: {'conditional': (10, -3, 'left'), 'B3': (-9, 9, 'right'), 'RAQoS_independent': (-2, 12, 'left'), 'fixed_half': (12, -1, 'left'), 'constant_sigmoid': (12, 1, 'left'), 'B3_nonlinear': (40, -2, 'left')}, 5: {'conditional': (10, -3, 'left'), 'B3': (-8, 9, 'right'), 'RAQoS_independent': (9, 3, 'left'), 'fixed_half': (15, 10, 'left'), 'constant_sigmoid': (15, -12, 'left'), 'B3_nonlinear': (0, 14, 'center')}}
    annotations = []
    coords_by_panel = []
    for (ax, density, letter) in zip(axes, (1, 5), ('a', 'b')):
        group = {r['method']: r for r in rows if int(r['density_percent']) == density}
        coords = {m: (float(r['mae_mean_s']), float(r['rmse_mean_s'])) for (m, r) in group.items()}
        coords_by_panel.append(coords)
        (xlim, ylim, xticks, yticks) = configs[density]
        ax.set(xlim=xlim, ylim=ylim, xlabel='MAE (s)', ylabel='RMSE (s)')
        ax.set_xticks([t for t in xticks if xlim[0] <= t <= xlim[1]])
        ax.set_yticks(yticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title(f'({letter}) RT {density}% training density', loc='left', pad=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=3, width=0.6)
        ax.grid(color='#E5E7EB', linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        assert all((xlim[0] < coords[m][0] < xlim[1] and ylim[0] < coords[m][1] < ylim[1] for m in METHODS))
        assert all((p < f for (p, f) in zip(coords['conditional'], coords['fixed_half'])))
        ax.annotate('', xy=coords['conditional'], xytext=coords['fixed_half'], arrowprops=dict(arrowstyle='-|>', color='#A75C79', alpha=0.65, lw=1.0, mutation_scale=9, shrinkA=5, shrinkB=7), zorder=2)
        for method in ('constant_sigmoid', 'fixed_half', 'B3_nonlinear', 'RAQoS_independent', 'B3', 'conditional'):
            (name, color, marker, size) = METHODS[method]
            (x, y) = coords[method]
            if method in EXPECTED_PARETO[density]:
                ax.scatter([x], [y], s=size + 43, marker=marker, facecolors='white', edgecolors='#7A8492', linewidths=0.55, zorder=4)
            hollow = method == 'constant_sigmoid'
            ax.scatter([x], [y], s=size, marker=marker, facecolors='none' if hollow else color, edgecolors=color, linewidths=1.1 if hollow else 0.55, zorder=5 if method != 'conditional' else 6)
            (dx, dy, ha) = labels[density][method]
            text = 'B3 nonlinear\ncalibration' if method == 'B3_nonlinear' else name
            leader = dict(arrowstyle='-', color='#A3AAB4', lw=0.55, shrinkA=2, shrinkB=5)
            ann = ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va='center', fontsize=7.5, color=color if method in ('conditional', 'B3', 'RAQoS_independent') else '#545E6C', fontweight='bold' if method == 'conditional' else 'normal', arrowprops=leader if method in ('fixed_half', 'constant_sigmoid', 'B3_nonlinear') else None, zorder=7, annotation_clip=False)
            annotations.append(ann)
    handles = [Line2D([], [], linestyle='none', marker=marker, markersize=math.sqrt(size), markerfacecolor='none' if m == 'constant_sigmoid' else color, markeredgecolor=color, markeredgewidth=1.0 if m == 'constant_sigmoid' else 0.6, label=name) for (m, (name, color, marker, size)) in METHODS.items()]
    order = [0, 3, 1, 4, 2, 5]
    legend = fig.legend(handles=[handles[i] for i in order], loc='lower center', bbox_to_anchor=(0.53, 0.04), ncol=3, frameon=False, fontsize=7.5, columnspacing=2.8, handletextpad=0.65, labelspacing=0.75)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [*annotations, *fig.texts, *legend.get_texts()]
    for ax in axes:
        texts.extend([ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()])
    for label in texts:
        bb = label.get_window_extent(renderer)
        assert bb.x0 >= 0 and bb.y0 >= 0 and (bb.x1 <= fig.bbox.width) and (bb.y1 <= fig.bbox.height), label.get_text()
    for (index, (ax, coords)) in enumerate(zip(axes, coords_by_panel)):
        panel_labels = annotations[index * 6:(index + 1) * 6]
        boxes = [Text.get_window_extent(label, renderer) for label in panel_labels]
        for (i, box) in enumerate(boxes):
            for j in range(i):
                assert not box.overlaps(boxes[j]), (panel_labels[i].get_text(), panel_labels[j].get_text())
            for (method, (_, _, _, size)) in METHODS.items():
                center = ax.transData.transform(coords[method])
                area = size + 43 if method in EXPECTED_PARETO[(1, 5)[index]] else size
                radius = (math.sqrt(area) / 2 + 0.5) * fig.dpi / 72
                marker_box = Bbox.from_extents(center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
                assert not box.overlaps(marker_box), (panel_labels[i].get_text(), method)
    stem = HERE / 'fig03_performance_tradeoff'
    fig.savefig(stem.with_suffix('.pdf'), metadata={'Title': 'MAE-RMSE performance trade-off', 'Author': 'Paper5'})
    fig.savefig(stem.with_suffix('.svg'), metadata={'Title': 'MAE-RMSE performance trade-off'})
    fig.savefig(stem.with_suffix('.png'), dpi=600)
    fig.savefig(HERE / 'fig03_preview_7p16in.png', dpi=150)
    for (ax, coords, density) in zip(axes, coords_by_panel, (1, 5)):
        distance = math.dist(ax.transData.transform(coords['fixed_half']), ax.transData.transform(coords['conditional']))
        print(f'RT {density}% Fixed-to-PCF center distance: {distance * 72 / fig.dpi:.2f} pt')
    plt.close(fig)
    print('Saved one 7.16 x 3.58 in candidate: vector PDF/SVG, 600-dpi PNG, 150-dpi preview.')

if __name__ == '__main__':
    plot()
