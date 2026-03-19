"""
Combined robustness figure: prompt variants + reward-function robustness.
Two-panel, NMI full-width (~7 in).
  Panel (a): Prompt variants (Tetris, Qwen-32B) + DFS baseline
  Panel (b): Reward robustness — overlay 3 reward functions (DFS + LLM)
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Reuse data loaders from plot_appendix_figures
sys.path.insert(0, OUT_DIR)
from plot_appendix_figures import load_tetris_prompt_data, load_reward_data


def setup_rcparams():
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
    })


def plot_combined_robustness():
    setup_rcparams()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7, 3.2))

    # ---- Panel (a): Prompt variants ----
    dfs_by_bw, prompt_data, budgets = load_tetris_prompt_data()
    x = np.arange(len(budgets))

    # DFS baseline
    dfs_means = [np.mean(dfs_by_bw[b]) for b in budgets]
    dfs_se = [np.std(dfs_by_bw[b]) / np.sqrt(len(dfs_by_bw[b]))
              if len(dfs_by_bw[b]) > 1 else 0 for b in budgets]
    ax_a.errorbar(x, dfs_means, yerr=dfs_se, fmt='o-', color='#333333',
                  linewidth=1.5, markersize=4, capsize=2, capthick=0.7,
                  label='DFS (base)', markeredgecolor='white',
                  markeredgewidth=0.3, zorder=10)

    # Prompt variants
    prompt_styles = {
        'minimal':  {'color': '#B2182B', 'marker': 'D', 'label': 'LLM (minimal)'},
        'cot':      {'color': '#2166AC', 'marker': 's', 'label': 'LLM (chain-of-thought)'},
        'expert':   {'color': '#F4A582', 'marker': '^', 'label': 'LLM (expert)'},
        'standard': {'color': '#92C5DE', 'marker': 'v', 'label': 'LLM (standard)'},
    }
    order = ['minimal', 'cot', 'expert', 'standard']

    for prompt in order:
        if prompt not in prompt_data:
            continue
        by_bw = prompt_data[prompt]
        p_budgets = sorted(by_bw.keys())
        p_x = [list(budgets).index(b) for b in p_budgets if b in budgets]
        means = [np.mean(by_bw[b]) for b in p_budgets if b in budgets]
        se = [np.std(by_bw[b]) / np.sqrt(len(by_bw[b]))
              if len(by_bw[b]) > 1 else 0 for b in p_budgets if b in budgets]
        style = prompt_styles[prompt]
        ax_a.errorbar(p_x, means, yerr=se, fmt=f'{style["marker"]}-',
                      color=style['color'], linewidth=1.0, markersize=3.5,
                      capsize=1.5, capthick=0.5, label=style['label'],
                      markeredgecolor='white', markeredgewidth=0.2)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([str(b) for b in budgets])
    ax_a.set_xlabel(r'$\mathcal{B}$ (beam width)')
    ax_a.set_ylabel(r'$J$ (lines cleared)')
    ax_a.set_title('a', loc='left', fontweight='bold')
    ax_a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), framealpha=0.9, ncol=2)
    ax_a.grid(True, alpha=0.2, linewidth=0.5)

    # ---- Panel (b): Reward robustness (overlaid) ----
    dfs_data, llm_data, r_budgets, reward_fns = load_reward_data()
    x_r = np.arange(len(r_budgets))

    # Colors and line styles for each reward function
    dfs_styles = {
        'aggressive':   {'color': '#333333', 'ls': '-',  'label': 'DFS (aggressive)'},
        'conservative': {'color': '#666666', 'ls': '--', 'label': 'DFS (conservative)'},
        'default':      {'color': '#999999', 'ls': ':',  'label': 'DFS (default)'},
    }
    llm_styles = {
        'aggressive':   {'color': '#B2182B', 'ls': '-',  'label': 'LLM (aggressive)'},
        'conservative': {'color': '#D6604D', 'ls': '--', 'label': 'LLM (conservative)'},
        'default':      {'color': '#F4A582', 'ls': ':',  'label': 'LLM (default)'},
    }

    # Plot all DFS first, then all LLM
    for rf in reward_fns:
        dfs_bw = dfs_data[rf]
        dfs_means = [np.mean(dfs_bw[b]) for b in r_budgets]
        dfs_se = [np.std(dfs_bw[b]) / np.sqrt(len(dfs_bw[b]))
                  if len(dfs_bw[b]) > 1 else 0 for b in r_budgets]
        ds = dfs_styles[rf]
        ax_b.errorbar(x_r, dfs_means, yerr=dfs_se, fmt='o',
                      color=ds['color'], linestyle=ds['ls'],
                      linewidth=1.2, markersize=3.5, capsize=1.5,
                      capthick=0.5, label=ds['label'],
                      markeredgecolor='white', markeredgewidth=0.3,
                      zorder=10)

    for rf in reward_fns:
        if rf in llm_data:
            llm_bw = llm_data[rf]
            llm_means = [np.mean(llm_bw[b]) for b in r_budgets if b in llm_bw]
            llm_se = [np.std(llm_bw[b]) / np.sqrt(len(llm_bw[b]))
                      if len(llm_bw[b]) > 1 else 0
                      for b in r_budgets if b in llm_bw]
            llm_x = [i for i, b in enumerate(r_budgets) if b in llm_bw]
            ls = llm_styles[rf]
            ax_b.errorbar(llm_x, llm_means, yerr=llm_se, fmt='s',
                          color=ls['color'], linestyle=ls['ls'],
                          linewidth=1.0, markersize=3, capsize=1.5,
                          capthick=0.5, label=ls['label'],
                          markeredgecolor='white', markeredgewidth=0.2)

    # Reorder legend: DFS left column, LLM right column (2 cols x 3 rows)
    handles, labels = ax_b.get_legend_handles_labels()
    # handles order: DFS_agg(0), DFS_cons(1), DFS_def(2), LLM_agg(3), LLM_cons(4), LLM_def(5)
    # For ncol=2 row-major: need [DFS_agg, LLM_agg, DFS_cons, LLM_cons, DFS_def, LLM_def]
    reorder = [0, 3, 1, 4, 2, 5]
    ax_b.legend([handles[i] for i in reorder], [labels[i] for i in reorder],
                loc='upper center', bbox_to_anchor=(0.5, -0.15), framealpha=0.9, ncol=2)

    ax_b.set_xticks(x_r)
    ax_b.set_xticklabels([str(b) for b in r_budgets])
    ax_b.set_xlabel(r'$\mathcal{B}$ (beam width)')
    ax_b.set_ylabel(r'$J$ (lines cleared)')
    ax_b.set_title('b', loc='left', fontweight='bold')
    ax_b.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout(pad=0.4, w_pad=1.0)
    for ext in ['pdf', 'png']:
        path = os.path.join(OUT_DIR, f'figure_robustness.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    plot_combined_robustness()
