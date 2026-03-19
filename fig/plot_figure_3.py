"""
Figure 3: AIME -- Delta comparison: nested derive (var+var) vs fixed derive average (var+const).
X-axis = generator model size, Y-axis = Delta averaged over k values.

nested derive: gen=sel=same model (diagonal of var+const matrix)
fixed derive:  average over all 5 selectors

Also generates figure_3_60 using only AIME 2024+2025 (60 problems).

PRL single-column format (3.375 in wide).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALING_DIR = os.path.join(BASE_DIR, 'scaling', 'results')

FIG_WIDTH = 3.375
FIG_HEIGHT = 3.0

MODELS = ['qwen-7b', 'qwen-14b', 'qwen-32b', 'qwen-72b', 'qwen3-max']
MODEL_LABELS = ['7B', '14B', '32B', '72B', 'Qwen3']
SELECTORS = MODELS  # all 5 as fixed selectors


def load_data(problem_types=None):
    """Load AIME data. If problem_types is set, filter to those types only."""
    with open(os.path.join(SCALING_DIR, 'phase3_results.json')) as f:
        varvar = json.load(f)
    with open(os.path.join(SCALING_DIR, 'phase3_lechatelier.json')) as f:
        lc = json.load(f)

    if problem_types:
        varvar = [r for r in varvar if r.get('problem_type') in problem_types]
        lc = [r for r in lc if r.get('problem_type') in problem_types]

    k_values = sorted(k for k in set(r['k'] for r in varvar) if k != 1)

    # MV accuracy per model per k
    mv_by_mk = {}
    for model in MODELS:
        mv_by_mk[model] = {}
        for k in k_values:
            recs = [r for r in varvar if r['method'] == 'majority_vote'
                    and r['model'] == model and r['k'] == k]
            mv_by_mk[model][k] = sum(r['correct'] for r in recs) / len(recs) if recs else 0

    # var+const agent
    vc = [r for r in lc if r['config'] == 'var+const']
    ag_varconst = {}
    for sel in SELECTORS:
        ag_varconst[sel] = {}
        for gen in MODELS:
            ag_varconst[sel][gen] = {}
            for k in k_values:
                recs = [r for r in vc if r['method'] == 'agent'
                        and r['generator'] == gen and r['selector'] == sel and r['k'] == k]
                ag_varconst[sel][gen][k] = sum(r['correct'] for r in recs) / len(recs) if recs else 0

    return k_values, mv_by_mk, ag_varconst


def plot_figure3(k_values, mv_by_mk, ag_vc, outdir, suffix=''):
    """Generate figure 3 (Delta) and figure 3 supp (raw J)."""
    all_mv = [mv_by_mk[m][k] for m in MODELS for k in k_values]
    avg_mv = np.mean(all_mv)
    x = np.arange(len(MODELS))

    # nested derived = diagonal of var+const, k=15,17,19,21
    K_SHOW = [15, 17, 19, 21]
    delta_nested, se_nested = [], []
    for model in MODELS:
        ds = [(mv_by_mk[model][k] - ag_vc[model][model][k]) / abs(avg_mv)
              for k in K_SHOW if avg_mv != 0]
        delta_nested.append(np.mean(ds))
        se_nested.append(np.std(ds) / np.sqrt(len(ds)) if len(ds) > 1 else 0)

    # fixed derived = average over all 5 selectors, k=15,17,19,21
    delta_fixed, se_fixed = [], []
    for gen in MODELS:
        ds_all = [(mv_by_mk[gen][k] - ag_vc[sel][gen][k]) / abs(avg_mv)
                  for sel in SELECTORS for k in K_SHOW if avg_mv != 0]
        delta_fixed.append(np.mean(ds_all))
        se_fixed.append(np.std(ds_all) / np.sqrt(len(ds_all)) if len(ds_all) > 1 else 0)

    # ========== Main figure: Delta ==========
    plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'legend.fontsize': 5.5,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
    })

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.errorbar(x, delta_nested, yerr=se_nested, fmt='o-', color='#B2182B',
                linewidth=1.5, markersize=4, capsize=2, capthick=0.7,
                label=r'nested derived', markeredgecolor='white', markeredgewidth=0.3,
                zorder=10)
    ax.errorbar(x, delta_fixed, yerr=se_fixed, fmt='s-', color='#2166AC',
                linewidth=1.5, markersize=4, capsize=2, capthick=0.7,
                label=r'fixed derived',
                markeredgecolor='white', markeredgewidth=0.3,
                zorder=9)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_xlabel('Generator model size')
    ax.set_ylabel(r'$\langle\Delta\rangle_k$')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout(pad=0.3)
    for ext in ['pdf', 'png']:
        outpath = os.path.join(outdir, f'figure_3{suffix}.{ext}')
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f'Saved {outpath}')
    plt.close(fig)

    # ========== Supp figure: raw J values ==========
    fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # MV line (k=15,17,19,21 avg)
    K_SUPP = [15, 17, 19, 21]
    j_mv = [np.mean([mv_by_mk[model][k] for k in K_SUPP]) for model in MODELS]
    ax2.plot(x, j_mv, 'o-', color='#333333',
                 linewidth=1.2, markersize=4,
                 label=r'MV (each model)', markeredgecolor='white', markeredgewidth=0.3,
                 zorder=10)

    # nested derived = diagonal (k=15,17,19,21 avg)
    j_nested = [np.mean([ag_vc[model][model][k] for k in K_SUPP]) for model in MODELS]
    ax2.plot(x, j_nested, 'o-', color='#B2182B',
             linewidth=1.2, markersize=4,
             label=r'Agent (nested derived)', markeredgecolor='white', markeredgewidth=0.3,
             zorder=9)

    # Shaded region below nested derive line
    y_bottom = ax2.get_ylim()[0]
    ax2.fill_between(x, y_bottom, j_nested, color='#B2182B', alpha=0.12, zorder=1)

    # 5 fixed derive agent lines
    sel_colors = ['#9ECAE1', '#6BAED6', '#3182BD', '#2171B5', '#084594']
    sel_markers = ['v', '^', 's', 'D', 'p']
    sel_labels = ['sel=7B', 'sel=14B', 'sel=32B', 'sel=72B', 'sel=Qwen3']
    for i, (sel, label) in enumerate(zip(SELECTORS, sel_labels)):
        j_sel = [np.mean([ag_vc[sel][gen][k] for k in K_SUPP]) for gen in MODELS]
        ax2.plot(x, j_sel, f'{sel_markers[i]}-', color=sel_colors[i],
                     linewidth=0.8, markersize=3,
                     label=f'Agent ({label})',
                     markeredgecolor='white', markeredgewidth=0.2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(MODEL_LABELS)
    ax2.set_xlabel('Generator model size')
    ax2.set_ylabel(r'$\langle J \rangle_{k}$ (accuracy)')
    ax2.legend(loc='best', framealpha=0.9, ncol=2, columnspacing=0.8,
               handletextpad=0.3, fontsize=5)
    ax2.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout(pad=0.3)
    for ext in ['pdf', 'png']:
        outpath = os.path.join(outdir, f'figure_3{suffix}_supp.{ext}')
        fig2.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f'Saved {outpath}')
    plt.close(fig2)


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))

    # Full dataset (90 problems: 2024+2025+2026)
    k_values, mv_by_mk, ag_vc = load_data()
    plot_figure3(k_values, mv_by_mk, ag_vc, outdir)

    # 60-problem subset (AIME 2024+2025 only)
    k_values_60, mv_60, ag_vc_60 = load_data(problem_types={'aime_2024', 'aime_2025'})
    plot_figure3(k_values_60, mv_60, ag_vc_60, outdir, suffix='_60')


if __name__ == '__main__':
    main()
