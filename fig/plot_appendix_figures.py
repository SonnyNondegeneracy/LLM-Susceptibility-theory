"""
Appendix figures for PRL paper:
  figure_prompt.pdf  — Prompt robustness (Tetris only, Qwen-32B)
  figure_reward.pdf  — Reward-function robustness (Tetris, Qwen-32B, 3 reward fns)

PRL single-column format (3.375 in wide).
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
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FIG_WIDTH = 3.375


def setup_rcparams():
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


def load_tetris_prompt_data():
    """Load DFS baseline + all prompt variant LLM data for Tetris."""
    with open(os.path.join(RESULTS_DIR, 'dfs_baseline.json')) as f:
        dfs_all = json.load(f)
    # Use aggressive reward function (matches main text)
    dfs = [r for r in dfs_all if r.get('reward_fn') == 'aggressive']

    dfs_by_bw = defaultdict(list)
    for r in dfs:
        dfs_by_bw[r['beam_width']].append(r['lines_cleared'])

    # Load all prompt variants for qwen-32b
    prompt_data = {}
    for prompt in ['minimal', 'standard', 'cot', 'expert']:
        fname = os.path.join(RESULTS_DIR, f'tetris_prompt_{prompt}.json')
        if os.path.exists(fname):
            with open(fname) as f:
                data = json.load(f)
            by_bw = defaultdict(list)
            for r in data:
                by_bw[r['beam_width']].append(r['lines_cleared'])
            prompt_data[prompt] = by_bw
        else:
            # Try the standard llm file for 'standard' prompt
            fname2 = os.path.join(RESULTS_DIR, 'llm_qwen-32b_aggressive.json')
            if prompt == 'standard' and os.path.exists(fname2):
                with open(fname2) as f:
                    data = json.load(f)
                by_bw = defaultdict(list)
                for r in data:
                    by_bw[r['beam_width']].append(r['lines_cleared'])
                prompt_data[prompt] = by_bw

    budgets = sorted(dfs_by_bw.keys())
    return dfs_by_bw, prompt_data, budgets


def plot_prompt_tetris():
    """Figure: prompt comparison, Tetris only, PRL format."""
    setup_rcparams()
    dfs_by_bw, prompt_data, budgets = load_tetris_prompt_data()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.8))

    x = np.arange(len(budgets))

    # DFS baseline
    dfs_means = [np.mean(dfs_by_bw[b]) for b in budgets]
    dfs_se = [np.std(dfs_by_bw[b]) / np.sqrt(len(dfs_by_bw[b]))
              if len(dfs_by_bw[b]) > 1 else 0 for b in budgets]
    ax.errorbar(x, dfs_means, yerr=dfs_se, fmt='o-', color='#333333',
                linewidth=1.5, markersize=4, capsize=2, capthick=0.7,
                label='DFS (base)', markeredgecolor='white', markeredgewidth=0.3,
                zorder=10)

    # Prompt variants
    prompt_styles = {
        'minimal': {'color': '#B2182B', 'marker': 'D', 'label': 'LLM (minimal)'},
        'cot':     {'color': '#2166AC', 'marker': 's', 'label': 'LLM (chain-of-thought)'},
        'expert':  {'color': '#F4A582', 'marker': '^', 'label': 'LLM (expert)'},
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
        ax.errorbar(p_x, means, yerr=se, fmt=f'{style["marker"]}-',
                    color=style['color'], linewidth=1.0, markersize=3.5,
                    capsize=1.5, capthick=0.5, label=style['label'],
                    markeredgecolor='white', markeredgewidth=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel(r'$\mathcal{B}$ (beam width)')
    ax.set_ylabel(r'$J$ (lines cleared)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout(pad=0.3)
    for ext in ['pdf', 'png']:
        path = os.path.join(OUT_DIR, f'figure_prompt.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)


def load_reward_data():
    """Load DFS baseline and LLM data for all 3 reward functions."""
    with open(os.path.join(RESULTS_DIR, 'dfs_baseline.json')) as f:
        dfs_all = json.load(f)

    reward_fns = ['aggressive', 'conservative', 'default']
    dfs_data = {}
    for rf in reward_fns:
        by_bw = defaultdict(list)
        for r in dfs_all:
            if r.get('reward_fn') == rf:
                by_bw[r['beam_width']].append(r['lines_cleared'])
        dfs_data[rf] = by_bw

    llm_data = {}
    for rf in reward_fns:
        fname = os.path.join(RESULTS_DIR, f'llm_qwen-32b_{rf}.json')
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            data = json.load(f)
        by_bw = defaultdict(list)
        for r in data:
            by_bw[r['beam_width']].append(r['lines_cleared'])
        llm_data[rf] = by_bw

    budgets = sorted(dfs_data['aggressive'].keys())
    return dfs_data, llm_data, budgets, reward_fns


def plot_reward_robustness():
    """Figure: reward-function robustness, 3 panels, PRL format."""
    setup_rcparams()
    dfs_data, llm_data, budgets, reward_fns = load_reward_data()

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH * 2, 2.4), sharey=True)
    titles = {'aggressive': '(a) Aggressive', 'conservative': '(b) Conservative',
              'default': '(c) Default'}

    for idx, rf in enumerate(reward_fns):
        ax = axes[idx]
        x = np.arange(len(budgets))

        # DFS
        dfs_bw = dfs_data[rf]
        dfs_means = [np.mean(dfs_bw[b]) for b in budgets]
        dfs_se = [np.std(dfs_bw[b]) / np.sqrt(len(dfs_bw[b]))
                  if len(dfs_bw[b]) > 1 else 0 for b in budgets]
        ax.errorbar(x, dfs_means, yerr=dfs_se, fmt='o-', color='#333333',
                    linewidth=1.2, markersize=3.5, capsize=1.5, capthick=0.5,
                    label='DFS', markeredgecolor='white', markeredgewidth=0.3,
                    zorder=10)

        # LLM
        if rf in llm_data:
            llm_bw = llm_data[rf]
            llm_means = [np.mean(llm_bw[b]) for b in budgets if b in llm_bw]
            llm_se = [np.std(llm_bw[b]) / np.sqrt(len(llm_bw[b]))
                      if len(llm_bw[b]) > 1 else 0 for b in budgets if b in llm_bw]
            llm_x = [i for i, b in enumerate(budgets) if b in llm_bw]
            ax.errorbar(llm_x, llm_means, yerr=llm_se, fmt='s-', color='#B2182B',
                        linewidth=1.0, markersize=3, capsize=1.5, capthick=0.5,
                        label='LLM (32B)', markeredgecolor='white',
                        markeredgewidth=0.2)

        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel(r'$\mathcal{B}$ (beam width)')
        ax.set_title(titles[rf])
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=5)
        if idx == 0:
            ax.set_ylabel(r'$J$ (lines cleared)')

    plt.tight_layout(pad=0.3, w_pad=0.5)
    for ext in ['pdf', 'png']:
        path = os.path.join(OUT_DIR, f'figure_reward.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    plot_prompt_tetris()
    plot_reward_robustness()
