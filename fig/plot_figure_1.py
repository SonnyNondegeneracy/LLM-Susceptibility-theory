"""
Figure 1: Tetris — Base algorithm vs LLM-derived strategy performance.
Each LLM model plotted individually. Points only + linear fits.
PRL single-column format (3.375 in wide).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

FIG_WIDTH = 3.375
FIG_HEIGHT = 2.8

LLM_MODELS = ['qwen-7b', 'qwen-14b', 'qwen-32b', 'qwen-72b', 'qwen3-max']
MODEL_LABELS = ['7B', '14B', '32B', '72B', 'Qwen3']
# Color gradient: light to dark red for increasing model size
MODEL_COLORS = ['#FCBBA1', '#FB6A4A', '#CB181D', '#99000D', '#67000D']
MODEL_MARKERS = ['v', '^', 's', 'D', 'p']


def load_tetris_data():
    with open(os.path.join(RESULTS_DIR, 'dfs_baseline.json')) as f:
        dfs = json.load(f)
    dfs_by_bw = defaultdict(list)
    for r in dfs:
        dfs_by_bw[r['beam_width']].append(r['lines_cleared'])

    # Per-model data
    model_data = {}
    for model in LLM_MODELS:
        fname = os.path.join(RESULTS_DIR, f'llm_{model}_aggressive.json')
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            data = json.load(f)
        by_bw = defaultdict(list)
        for r in data:
            by_bw[r['beam_width']].append(r['lines_cleared'])
        model_data[model] = by_bw

    return dfs_by_bw, model_data


def main():
    dfs_by_bw, model_data = load_tetris_data()
    beam_widths = sorted(dfs_by_bw.keys())
    x = np.arange(len(beam_widths))

    dfs_means = np.array([np.mean(dfs_by_bw[bw]) for bw in beam_widths])
    dfs_se = np.array([np.std(dfs_by_bw[bw]) / np.sqrt(len(dfs_by_bw[bw])) for bw in beam_widths])

    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 5.5,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
    })

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # DFS data points
    ax.errorbar(x, dfs_means, yerr=dfs_se, fmt='o', color='#2166AC',
                markersize=5, capsize=2, capthick=0.8,
                label=r'$\mathcal{P}_\mathcal{B}$ (DFS)', zorder=5,
                markeredgecolor='white', markeredgewidth=0.4)

    # DFS linear fit
    slope_dfs, intercept_dfs = np.polyfit(x, dfs_means, 1)
    x_fit = np.linspace(x[0] - 0.3, x[-1] + 0.3, 100)
    ax.plot(x_fit, slope_dfs * x_fit + intercept_dfs, '--', color='#2166AC',
            linewidth=1.0, alpha=0.7, zorder=2, label='DFS fit')

    # Each LLM model: data points
    all_llm_means = []
    for i, (model, label) in enumerate(zip(LLM_MODELS, MODEL_LABELS)):
        if model not in model_data:
            continue
        by_bw = model_data[model]
        means = np.array([np.mean(by_bw[bw]) for bw in beam_widths])
        se = np.array([np.std(by_bw[bw]) / np.sqrt(len(by_bw[bw])) for bw in beam_widths])
        all_llm_means.append(means)

        ax.errorbar(x, means, yerr=se, fmt=MODEL_MARKERS[i],
                    color=MODEL_COLORS[i], markersize=4, capsize=1.5, capthick=0.5,
                    label=f"$\\mathcal{{P}}'_\\mathcal{{B}}$ ({label})", zorder=4,
                    markeredgecolor='white', markeredgewidth=0.3)

    # Averaged LLM linear fit (one dashed line)
    avg_llm = np.mean(all_llm_means, axis=0)
    slope_llm, intercept_llm = np.polyfit(x, avg_llm, 1)
    ax.plot(x_fit, slope_llm * x_fit + intercept_llm, '--', color='#B2182B',
            linewidth=1.0, alpha=0.7, zorder=2,
            label=f'LLM avg fit')

    # Slope annotations
    # ax.text(x[-1] + 0.2, slope_dfs * (x[-1] + 0.2) + intercept_dfs + 0.4,
    #         f'{slope_dfs:.1f}', fontsize=6, color='#2166AC', va='bottom')
    # ax.text(x[-1] + 0.2, slope_llm * (x[-1] + 0.2) + intercept_llm - 0.4,
    #         f'{slope_llm:.1f}', fontsize=6, color='#B2182B', va='top')

    ax.set_xticks(x)
    ax.set_xticklabels([str(bw) for bw in beam_widths])
    ax.set_xlabel(r'Computational budget $\mathcal{B}$ (beam width)')
    ax.set_ylabel(r'Performance $J$ (lines cleared)')
    ax.legend(loc='upper left', framealpha=0.9, ncol=2, columnspacing=0.8,
              handletextpad=0.3)
    ax.set_ylim(-1, 14.5)
    ax.set_xlim(-0.4, x[-1] + 0.4)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout(pad=0.3)

    outdir = os.path.dirname(os.path.abspath(__file__))
    for ext in ['pdf', 'png']:
        outpath = os.path.join(outdir, f'figure_1.{ext}')
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f'Saved {outpath}')
    plt.close(fig)


if __name__ == '__main__':
    main()
