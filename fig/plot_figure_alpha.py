"""
Figure: Responsiveness α(k) for AIME (60 problems, 2024+2025).

α = ⟨∂J(P'_B)/∂J(P_B)⟩  where B = model capability.

For each k, plot J_agent vs J_MV across 5 models → slope = α(k).
X-axis = k, Y-axis = α(k). α ≤ 1 verifies Le Chatelier's principle.
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
SELECTORS = MODELS

AIME_PROBLEM_TYPES = {'aime_2024', 'aime_2025'}


def load_data():
    with open(os.path.join(SCALING_DIR, 'phase3_results.json')) as f:
        varvar = json.load(f)
    with open(os.path.join(SCALING_DIR, 'phase3_lechatelier.json')) as f:
        lc = json.load(f)

    varvar = [r for r in varvar if r.get('problem_type') in AIME_PROBLEM_TYPES]
    lc = [r for r in lc if r.get('problem_type') in AIME_PROBLEM_TYPES]

    k_values = sorted(k for k in set(r['k'] for r in varvar) if k != 1)

    # Only use k values that exist in both datasets
    lc_ks = set(r['k'] for r in lc)
    k_values = [k for k in k_values if k in lc_ks]

    # MV accuracy per model per k
    mv = {}
    for model in MODELS:
        mv[model] = {}
        for k in k_values:
            recs = [r for r in varvar if r['method'] == 'majority_vote'
                    and r['model'] == model and r['k'] == k]
            mv[model][k] = sum(r['correct'] for r in recs) / len(recs) if recs else 0

    # var+const agent: avg over all selectors
    vc = [r for r in lc if r['config'] == 'var+const']
    ag = {}
    for gen in MODELS:
        ag[gen] = {}
        for k in k_values:
            vals = []
            for sel in SELECTORS:
                recs = [r for r in vc if r['method'] == 'agent'
                        and r['generator'] == gen and r['selector'] == sel and r['k'] == k]
                if recs:
                    vals.append(sum(r['correct'] for r in recs) / len(recs))
            ag[gen][k] = np.mean(vals) if vals else 0

    return k_values, mv, ag


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    k_values, mv, ag = load_data()

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

    # For each k, compute α = slope of J_agent vs J_MV across 5 models (linear fit)
    alphas = []
    alpha_ses = []
    for k in k_values:
        j_mv = np.array([mv[model][k] for model in MODELS])
        j_ag = np.array([ag[model][k] for model in MODELS])

        # Linear fit: j_ag = α * j_mv + β
        # Use np.polyfit degree 1
        coeffs, cov = np.polyfit(j_mv, j_ag, 1, cov=True)
        alpha = coeffs[0]
        alpha_se = np.sqrt(cov[0, 0])
        alphas.append(alpha)
        alpha_ses.append(alpha_se)

    alphas = np.array(alphas)
    alpha_ses = np.array(alpha_ses)

    # ========== Main figure ==========
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.errorbar(k_values, alphas, yerr=alpha_ses, fmt='o-', color='#2166AC',
                linewidth=1.5, markersize=4, capsize=2, capthick=0.7,
                markeredgecolor='white', markeredgewidth=0.3,
                label=r'$\alpha(k)$ (linear fit)')

    ax.axhline(1, color='gray', linewidth=0.7, linestyle='--', alpha=0.6,
               label=r'$\alpha = 1$')
    ax.axhline(0, color='gray', linewidth=0.4, linestyle='-', alpha=0.3)

    # Shade α > 1 region
    ax.fill_between([k_values[0] - 1, k_values[-1] + 1], 1, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 2,
                    color='#B2182B', alpha=0.08)

    ax.set_xlabel(r'$k$ (number of samples)')
    ax.set_ylabel(r'$\alpha(k)$')
    ax.set_xticks(k_values)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout(pad=0.3)
    for ext in ['pdf', 'png']:
        path = os.path.join(outdir, f'figure_alpha.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)

    # Print values
    print("\nα(k) values:")
    for k, a, se in zip(k_values, alphas, alpha_ses):
        print(f"  k={k:2d}: α = {a:.3f} ± {se:.3f}")


if __name__ == '__main__':
    main()
