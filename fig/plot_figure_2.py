"""
Figure 2: Normalized advantage Δ(B) across four task domains.
Δ = [J(P_B) - J(P'_B)] / <J(P_B)>_B

Generates three outputs:
  figure_2.pdf       — averaged across LLM models
  figure_2_models.pdf — per-model results
  figure_2_supp.pdf   — J(P_B) vs B for each domain

PRL single-column format (3.375 in wide), 2x2 panels.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SCALING_DIR = os.path.join(BASE_DIR, 'scaling', 'results')

FIG_WIDTH = 3.375
FIG_HEIGHT = 3.6

LLM_MODELS = ['qwen-7b', 'qwen-14b', 'qwen-32b', 'qwen-72b', 'qwen3-max']
MODEL_LABELS = ['7B', '14B', '32B', '72B', 'Qwen3']
MODEL_COLORS = ['#FCBBA1', '#FB6A4A', '#CB181D', '#99000D', '#67000D']
MODEL_MARKERS = ['v', '^', 's', 'D', 'p']

AIME_MODELS = ['qwen-7b', 'qwen-14b', 'qwen-32b', 'qwen-72b', 'qwen3-max']


# ====================== Data Loaders ======================

def load_tetris():
    with open(os.path.join(RESULTS_DIR, 'dfs_baseline.json')) as f:
        dfs = json.load(f)
    dfs_by_bw = defaultdict(list)
    for r in dfs:
        dfs_by_bw[r['beam_width']].append(r['lines_cleared'])

    # All LLM pooled
    llm_all = defaultdict(list)
    # Per-model
    llm_per_model = {}
    for model in LLM_MODELS:
        fname = os.path.join(RESULTS_DIR, f'llm_{model}_aggressive.json')
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            data = json.load(f)
        by_bw = defaultdict(list)
        for r in data:
            by_bw[r['beam_width']].append(r['lines_cleared'])
            llm_all[r['beam_width']].append(r['lines_cleared'])
        llm_per_model[model] = by_bw

    budgets = sorted(dfs_by_bw.keys())
    return dfs_by_bw, llm_all, llm_per_model, budgets


def load_knapsack():
    with open(os.path.join(RESULTS_DIR, 'knapsack_v2_dfs_baseline.json')) as f:
        dfs = json.load(f)
    dfs_by_bw = defaultdict(list)
    for r in dfs:
        dfs_by_bw[r['beam_width']].append(r['total_value'])

    llm_all = defaultdict(list)
    llm_per_model = {}
    for model in LLM_MODELS:
        fname = os.path.join(RESULTS_DIR, f'knapsack_v2_llm_{model}_standard.json')
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            data = json.load(f)
        by_bw = defaultdict(list)
        for r in data:
            by_bw[r['beam_width']].append(r['total_value'])
            llm_all[r['beam_width']].append(r['total_value'])
        llm_per_model[model] = by_bw

    budgets = sorted(dfs_by_bw.keys())
    return dfs_by_bw, llm_all, llm_per_model, budgets


def load_ranking():
    with open(os.path.join(RESULTS_DIR, 'ranking_dfs_baseline.json')) as f:
        dfs = json.load(f)
    dfs_by_b = defaultdict(list)
    for r in dfs:
        dfs_by_b[r['budget']].append(float(r['correct']))

    llm_all = defaultdict(list)
    llm_per_model = {}
    for model in LLM_MODELS:
        fname = os.path.join(RESULTS_DIR, f'ranking_llm_{model}_standard.json')
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            data = json.load(f)
        by_b = defaultdict(list)
        for r in data:
            by_b[r['budget']].append(float(r['correct']))
            llm_all[r['budget']].append(float(r['correct']))
        llm_per_model[model] = by_b

    budgets = sorted(dfs_by_b.keys())
    return dfs_by_b, llm_all, llm_per_model, budgets


AIME_PROBLEM_TYPES = {'aime_2024', 'aime_2025'}  # 60 problems


def load_aime():
    """Load AIME data for panel (d), filtered to 2024+2025 (60 problems):
    Base P_B = each model's own majority vote (var+var, phase3_results.json)
    Derived P'_B = var+const agent per selector (phase3_lechatelier.json)
    X-axis = model size (generator).
    """
    with open(os.path.join(SCALING_DIR, 'phase3_results.json')) as f:
        varvar = json.load(f)
    varvar = [r for r in varvar if r.get('problem_type') in AIME_PROBLEM_TYPES]

    model_order = AIME_MODELS
    selectors = AIME_MODELS
    k_values = sorted(k for k in set(r['k'] for r in varvar) if k != 1)

    mv_by_model_k = {}
    for model in model_order:
        mv_by_model_k[model] = {}
        for k in k_values:
            recs = [r for r in varvar if r['method'] == 'majority_vote'
                    and r['model'] == model and r['k'] == k]
            mv_by_model_k[model][k] = sum(r['correct'] for r in recs) / len(recs) if recs else 0

    with open(os.path.join(SCALING_DIR, 'phase3_lechatelier.json')) as f:
        lc = json.load(f)
    lc = [r for r in lc if r.get('problem_type') in AIME_PROBLEM_TYPES]
    vc = [r for r in lc if r['config'] == 'var+const']

    ag_by_sel_gen_k = {}
    for sel in selectors:
        ag_by_sel_gen_k[sel] = {}
        for gen in model_order:
            ag_by_sel_gen_k[sel][gen] = {}
            for k in k_values:
                recs = [r for r in vc if r['method'] == 'agent'
                        and r['generator'] == gen and r['selector'] == sel and r['k'] == k]
                ag_by_sel_gen_k[sel][gen][k] = sum(r['correct'] for r in recs) / len(recs) if recs else 0

    return model_order, selectors, k_values, mv_by_model_k, ag_by_sel_gen_k


# ====================== Delta Computation ======================

def compute_delta_avg(base_by_b, llm_by_b, budgets):
    """Δ with pooled LLM data, denominator = <J(P_B)>_B."""
    base_means = [np.mean(base_by_b[b]) for b in budgets]
    llm_means = [np.mean(llm_by_b[b]) for b in budgets]
    avg_base = np.mean(base_means)

    deltas, se_list = [], []
    for i, b in enumerate(budgets):
        d = (base_means[i] - llm_means[i]) / abs(avg_base) if avg_base != 0 else 0
        deltas.append(d)
        n_b = len(base_by_b[b])
        n_l = len(llm_by_b[b])
        se_b = np.std(base_by_b[b]) / np.sqrt(n_b) if n_b > 1 else 0
        se_l = np.std(llm_by_b[b]) / np.sqrt(n_l) if n_l > 1 else 0
        se_list.append(np.sqrt(se_b**2 + se_l**2) / abs(avg_base) if avg_base != 0 else 0)
    return deltas, se_list, base_means, llm_means


def compute_delta_per_model(base_by_b, llm_model_by_b, budgets):
    """Δ per model with SE. Base is shared (DFS), denominator = <J_base>_B."""
    base_means = [np.mean(base_by_b[b]) for b in budgets]
    avg_base = np.mean(base_means)

    deltas, se_list = [], []
    for b in budgets:
        base_vals = base_by_b[b]
        llm_vals = llm_model_by_b.get(b, [])
        llm_mean = np.mean(llm_vals) if llm_vals else np.nan
        d = (np.mean(base_vals) - llm_mean) / abs(avg_base) if avg_base != 0 else 0
        deltas.append(d)
        n_b = len(base_vals)
        n_l = len(llm_vals) if llm_vals else 0
        se_b = np.std(base_vals) / np.sqrt(n_b) if n_b > 1 else 0
        se_l = np.std(llm_vals) / np.sqrt(n_l) if n_l > 1 else 0
        se_list.append(np.sqrt(se_b**2 + se_l**2) / abs(avg_base) if avg_base != 0 else 0)
    return deltas, se_list


# ====================== Plotting ======================

def setup_rcparams():
    plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 7.5,
        'axes.titlesize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 5,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
    })


def plot_delta_panel(ax, deltas, se, budgets, xlabel, title, marker='o'):
    x = np.arange(len(budgets))
    ax.errorbar(x, deltas, yerr=se, fmt=f'{marker}-', color='#2166AC',
                linewidth=1.2, markersize=3, capsize=1.5, capthick=0.6)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    d_arr = np.array(deltas)
    ax.fill_between(x, 0, deltas, where=d_arr >= 0,
                    color='#2166AC', alpha=0.15, interpolate=True)
    ax.fill_between(x, 0, deltas, where=d_arr < 0,
                    color='#B2182B', alpha=0.15, interpolate=True)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.15, linewidth=0.4)


def plot_delta_models_panel(ax, base_by_b, llm_per_model, budgets, xlabel, title,
                            model_list, labels, colors, markers):
    """Plot Δ for each model individually with error bars."""
    x = np.arange(len(budgets))
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')

    for i, (model, label) in enumerate(zip(model_list, labels)):
        if model not in llm_per_model:
            continue
        deltas, se = compute_delta_per_model(base_by_b, llm_per_model[model], budgets)
        ax.errorbar(x, deltas, yerr=se, fmt=f'{markers[i]}-', color=colors[i],
                    linewidth=0.8, markersize=2.5, capsize=1.5, capthick=0.5,
                    label=label, markeredgecolor='white', markeredgewidth=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.legend(fontsize=4.5, ncol=2, columnspacing=0.5, handletextpad=0.3)


def main():
    setup_rcparams()
    outdir = os.path.dirname(os.path.abspath(__file__))

    # Load all data
    tet_base, tet_llm, tet_per, tet_b = load_tetris()
    ks_base, ks_llm, ks_per, ks_b = load_knapsack()
    rk_base, rk_llm, rk_per, rk_b = load_ranking()
    ai_sels, ai_selectors, ai_ks, ai_mv_mk, ai_ag_sgk = load_aime()

    # ========== Figure 2: Averaged Δ(B) ==========
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

    tet_d, tet_se, tet_jb, _ = compute_delta_avg(tet_base, tet_llm, tet_b)
    ks_d, ks_se, ks_jb, _ = compute_delta_avg(ks_base, ks_llm, ks_b)
    rk_d, rk_se, rk_jb, _ = compute_delta_avg(rk_base, rk_llm, rk_b)

    plot_delta_panel(axes[0, 0], tet_d, tet_se, tet_b,
                     r'$\mathcal{B}$ (beam width)', '(a) Tetris', 'o')
    axes[0, 0].set_ylabel(r'$\Delta(\mathcal{B})$')

    plot_delta_panel(axes[0, 1], ks_d, ks_se, ks_b,
                     r'$\mathcal{B}$ (beam width)', '(b) 0/1 Knapsack', 's')

    plot_delta_panel(axes[1, 0], rk_d, rk_se, rk_b,
                     r'$\mathcal{B}$ (SNR)', '(c) Ranking', '^')
    axes[1, 0].set_ylabel(r'$\Delta(\mathcal{B})$')

    # Panel (d): AIME var+const — Δ vs generator model, k=9,15,21 avg
    # Base = each model's own MV (var+var), Derived = var+const agent (avg over all selectors)
    ax = axes[1, 1]
    x_ai = np.arange(len(ai_sels))
    K_AIME = [9, 15, 21]
    # <J(P_B)>_B: average MV across all models at k=9,15,21
    all_mv = [ai_mv_mk[m][k] for m in ai_sels for k in K_AIME]
    avg_mv = np.mean(all_mv)
    deltas_avg, se_avg = [], []
    for gen in ai_sels:
        ds = [(ai_mv_mk[gen][k] - ai_ag_sgk[sel][gen][k]) / abs(avg_mv)
              for sel in ai_selectors for k in K_AIME if avg_mv != 0]
        deltas_avg.append(np.mean(ds))
        se_avg.append(np.std(ds) / np.sqrt(len(ds)) if len(ds) > 1 else 0)
    ax.errorbar(x_ai, deltas_avg, yerr=se_avg, fmt='D-', color='#2166AC',
                linewidth=1.2, markersize=3, capsize=1.5, capthick=0.6)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    d_arr = np.array(deltas_avg)
    ax.fill_between(x_ai, 0, deltas_avg, where=d_arr >= 0,
                    color='#2166AC', alpha=0.15, interpolate=True)
    ax.set_xticks(x_ai)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_xlabel(r'Model size')
    ax.set_title('(d) AIME Math (fixed derive)')
    ax.grid(True, alpha=0.15, linewidth=0.4)

    plt.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.5)
    for ext in ['pdf', 'png']:
        path = os.path.join(outdir, f'figure_2.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)

    # ========== Figure 2 models: Per-model Δ(B) ==========
    fig2, axes2 = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Tetris per-model
    plot_delta_models_panel(axes2[0, 0], tet_base, tet_per, tet_b,
                            r'$\mathcal{B}$ (beam width)', '(a) Tetris',
                            LLM_MODELS, MODEL_LABELS, MODEL_COLORS, MODEL_MARKERS)
    axes2[0, 0].set_ylabel(r'$\Delta(\mathcal{B})$')

    # Knapsack per-model
    plot_delta_models_panel(axes2[0, 1], ks_base, ks_per, ks_b,
                            r'$\mathcal{B}$ (beam width)', '(b) 0/1 Knapsack',
                            LLM_MODELS, MODEL_LABELS, MODEL_COLORS, MODEL_MARKERS)

    # Ranking per-model
    plot_delta_models_panel(axes2[1, 0], rk_base, rk_per, rk_b,
                            r'$\mathcal{B}$ (SNR)', '(c) Ranking',
                            LLM_MODELS, MODEL_LABELS, MODEL_COLORS, MODEL_MARKERS)
    axes2[1, 0].set_ylabel(r'$\Delta(\mathcal{B})$')

    # AIME per-k: show Δ at each k value vs generator model
    ax = axes2[1, 1]
    x_ai = np.arange(len(ai_sels))
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    k_show = [3, 5, 9, 15, 21]
    k_colors = ['#C6DBEF', '#9ECAE1', '#6BAED6', '#3182BD', '#084594']
    k_markers = ['v', '^', 's', 'D', 'p']
    avg_mv = np.mean([ai_mv_mk[m][k] for m in ai_sels for k in ai_ks])
    for j, k in enumerate(k_show):
        deltas = [np.mean([(ai_mv_mk[gen][k] - ai_ag_sgk[sel][gen][k]) / abs(avg_mv)
                           for sel in ai_selectors])
                  for gen in ai_sels]
        ax.plot(x_ai, deltas, marker=k_markers[j], color=k_colors[j],
                linewidth=0.8, markersize=2.5, label=f'$k={k}$',
                markeredgecolor='white', markeredgewidth=0.2)
    ax.set_xticks(x_ai)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_xlabel('Model size')
    ax.set_title('(d) AIME Math (fixed derive)')
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.legend(fontsize=4.5, ncol=1, handletextpad=0.3)

    plt.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.5)
    for ext in ['pdf', 'png']:
        path = os.path.join(outdir, f'figure_2_models.{ext}')
        fig2.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig2)

    # ========== Supplementary: J(P_B) vs B ==========
    fig3, axes3 = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Compute means and SEs for base and LLM
    def j_means_se(data_by_b, budgets):
        means = [np.mean(data_by_b[b]) for b in budgets]
        ses = [np.std(data_by_b[b]) / np.sqrt(len(data_by_b[b]))
               if len(data_by_b[b]) > 1 else 0 for b in budgets]
        return means, ses

    tet_jb, tet_jb_se = j_means_se(tet_base, tet_b)
    tet_jl, tet_jl_se = j_means_se(tet_llm, tet_b)
    ks_jb, ks_jb_se = j_means_se(ks_base, ks_b)
    ks_jl, ks_jl_se = j_means_se(ks_llm, ks_b)
    rk_jb, rk_jb_se = j_means_se(rk_base, rk_b)
    rk_jl, rk_jl_se = j_means_se(rk_llm, rk_b)

    # AIME supp: J(MV) and J(Agent) at k=9,15,21 avg vs generator model
    # Each model's own MV vs var+const agent (averaged over selectors)
    K_SUPP = [9, 15, 21]
    ai_mv_vals = [np.mean([ai_mv_mk[gen][k] for k in K_SUPP]) for gen in ai_sels]
    ai_ag_vals = [np.mean([ai_ag_sgk[sel][gen][k] for sel in ai_selectors for k in K_SUPP])
                  for gen in ai_sels]
    # SE: binomial SE for MV (n=60 problems × 3 k), agent (n=60*5*3)
    ai_mv_se = [np.sqrt(p * (1 - p) / (60 * len(K_SUPP))) for p in ai_mv_vals]
    ai_ag_se = [np.sqrt(p * (1 - p) / (60 * len(ai_selectors) * len(K_SUPP))) for p in ai_ag_vals]

    datasets = [
        (tet_b, tet_jb, tet_jb_se, tet_jl, tet_jl_se, r'$\mathcal{B}$ (beam width)', '(a) Tetris', 'Lines cleared'),
        (ks_b, ks_jb, ks_jb_se, ks_jl, ks_jl_se, r'$\mathcal{B}$ (beam width)', '(b) 0/1 Knapsack', 'Total value'),
        (rk_b, rk_jb, rk_jb_se, rk_jl, rk_jl_se, r'$\mathcal{B}$ (SNR)', '(c) Ranking', 'Accuracy'),
        (MODEL_LABELS, ai_mv_vals, ai_mv_se, ai_ag_vals, ai_ag_se, 'Model size', '(d) AIME Math (fixed derive)', 'Accuracy'),
    ]

    for idx, (budgets, j_base, se_base, j_llm, se_llm, xlabel, title, ylabel) in enumerate(datasets):
        ax = axes3[idx // 2, idx % 2]
        x = np.arange(len(budgets))
        ax.errorbar(x, j_base, yerr=se_base, fmt='o-', color='#2166AC',
                    linewidth=1.2, markersize=3, capsize=1.5, capthick=0.6,
                    label=r'$J(\mathcal{P}_\mathcal{B})$')
        ax.errorbar(x, j_llm, yerr=se_llm, fmt='s-', color='#B2182B',
                    linewidth=1.2, markersize=3, capsize=1.5, capthick=0.6,
                    label=r"$J(\mathcal{P}'_\mathcal{B})$")
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        if idx % 2 == 0:
            ax.set_ylabel(ylabel)
        ax.legend(fontsize=5, loc='best')
        ax.grid(True, alpha=0.15, linewidth=0.4)

    plt.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.5)
    for ext in ['pdf', 'png']:
        path = os.path.join(outdir, f'figure_2_supp.{ext}')
        fig3.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig3)


if __name__ == '__main__':
    main()
