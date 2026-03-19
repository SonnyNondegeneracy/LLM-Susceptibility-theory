# A Theory of LLM Information Susceptibility

Code for the paper *"A Theory of LLM Information Susceptibility"* .

## Overview

We propose a theory of LLM information susceptibility: when computational resources are sufficiently large, the intervention of a fixed LLM does not increase the performance susceptibility of a strategy set with respect to budget. We validate the theory across four domains (Tetris, 0/1 Knapsack, world-knowledge Ranking, AIME mathematics) and five Qwen-series models (7B--236B), and show that nested, co-scaling architectures can exceed the fixed-layer bound.

## Repository Structure

```
.
├── game_engine.py                 # Tetris engine (10×20 board, heuristic evaluation, beam search)
├── llm_agent.py                   # LLM agent wrapper (DashScope API, multi-model)
├── requirements.txt
│
├── experiments/
│   ├── experiment.py              # Tetris: DFS baseline + LLM strategies + reward variants
│   ├── experiment_tetris_expand.py    # Expand Tetris from 10 to 40 seeds
│   ├── experiment_tetris_prompts.py   # Prompt variants (minimal/standard/cot/expert)
│   ├── experiment_knapsack_v2.py  # 0/1 Knapsack: beam search + LLM selection
│   └── experiment_ranking.py      # World-knowledge ranking under noise
│
├── scaling/                       # AIME scaling experiments (generator × selector)
│   ├── math_problems.py           # AIME 2024 problem definitions
│   ├── hard_math_problems.py      # AIME 2025 problem definitions
│   ├── experiment_phase3.py       # var+var (generator = selector)
│   ├── experiment_phase3_lechatelier.py  # var+const (fixed selector)
│   ├── experiment_varconst_all.py # var+const for all 5 selectors
│   └── experiment_add_k.py        # Add k=17,19 evaluations
│
└── fig/                           # Plotting scripts (one per paper figure)
    ├── plot_figure_1.py           # Fig. 1: Tetris J vs B
    ├── plot_figure_robustness.py  # Fig. 2: Prompt + reward robustness
    ├── plot_figure_2.py           # Fig. 3: Cross-domain validation
    ├── plot_figure_alpha.py       # Fig. 4: α(k) transition
    ├── plot_figure_response.py    # Fig. 4: Coupling regimes (schematic)
    ├── plot_figure_3.py           # Fig. 5: Nested vs fixed AIME
    └── plot_appendix_figures.py   # Extended Data figures
```

## Models

| Model | Parameters |
|-------|-----------|
| Qwen-2.5-7B-Instruct | 7B |
| Qwen-2.5-14B-Instruct | 14B |
| Qwen-2.5-32B-Instruct | 32B |
| Qwen-2.5-72B-Instruct | 72B |
| Qwen3-Max | ~236B |

## Domains

| Domain | Base strategy | Budget $\mathcal{B}$ | Performance $J$ |
|--------|--------------|------|-----------------|
| Tetris | Beam search with depth-first backtracking | Beam width | Lines cleared |
| 0/1 Knapsack | Beam search (value-density sort) | Beam width | Total value |
| Ranking | Noisy score estimation | Signal-to-noise ratio | Fraction correct |
| AIME Math | Majority vote over $k$ samples | Model size / $k$ | Accuracy |

## Setup

```bash
pip install -r requirements.txt
```

Create `api_key.json` in the project root with your DashScope API key:

```json
{
    "qwen-32b": [{"api_key": "YOUR_KEY", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen2.5-32b-instruct"}],
    "qwen-14b": [{"api_key": "YOUR_KEY", "base_url": "...", "model": "qwen2.5-14b-instruct"}],
    "qwen-7b":  [{"api_key": "YOUR_KEY", "base_url": "...", "model": "qwen2.5-7b-instruct"}],
    "qwen-72b": [{"api_key": "YOUR_KEY", "base_url": "...", "model": "qwen2.5-72b-instruct"}],
    "qwen3_new": [{"api_key": "YOUR_KEY", "base_url": "...", "model": "qwen3-max"}]
}
```

## Running Experiments

All experiments require a DashScope API key configured in `api_key.json` (see Setup). Results are saved to `results/` or `scaling/results/`.

### Tetris (Fig. 1, 2, 3)

```bash
# Step 1: DFS baseline + LLM strategies (5 models, aggressive reward)
#         Also runs reward variants (conservative, default) for Qwen-32B
python experiments/experiment.py                    # → results/dfs_baseline.json
                                                    # → results/llm_qwen-*_aggressive.json
                                                    # → results/llm_qwen-32b_{conservative,default}.json

# Step 2: Expand from 10 to 40 random seeds (merges into existing files)
python experiments/experiment_tetris_expand.py       # updates dfs_baseline.json & llm_*.json

# Step 3: Prompt variants (minimal, cot, expert) for Qwen-32B
python experiments/experiment_tetris_prompts.py      # → results/tetris_prompt_{minimal,cot,expert}.json
```

Flags for `experiment.py`:
- `--phase {dfs,llm,all}`: run DFS only, LLM only, or both (default: `all`)
- `--model MODEL`: run a single model (e.g., `qwen-32b`)
- `--reward REWARD`: reward function (default: `aggressive`)

### 0/1 Knapsack (Fig. 3)

```bash
python experiments/experiment_knapsack_v2.py         # → results/knapsack_v2_dfs_baseline.json
                                                     # → results/knapsack_v2_llm_*.json
```

Flags: `--phase {dfs,llm,all}`, `--model MODEL`

### World-knowledge Ranking (Fig. 3)

```bash
python experiments/experiment_ranking.py             # → results/ranking_dfs_baseline.json
                                                     # → results/ranking_llm_*.json
```

Flags: `--phase {dfs,llm,all}`, `--model MODEL`

### AIME Mathematics (Fig. 3, 4, 5)

The AIME experiments run in sequence; later steps reuse earlier generations.

```bash
cd scaling

# Step 1: Generate solutions & evaluate var+var (generator = selector)
python experiment_phase3.py                          # → results/phase3_generations.json
                                                     # → results/phase3_results.json

# Step 2: Evaluate var+const with 14B selector
python experiment_phase3_lechatelier.py              # → results/phase3_lechatelier.json

# Step 3: Extend var+const to all 5 selectors
python experiment_varconst_all.py                    # appends to phase3_lechatelier.json

# Step 4: Add k=17,19 evaluations (reuses existing generations)
python experiment_add_k.py                           # appends to phase3_results.json
                                                     #   & phase3_lechatelier.json
```

Flags for `experiment_phase3.py`:
- `--eval-only`: skip generation, evaluate from cached results
- `--models MODEL [MODEL ...]`: run a subset of models
- `--const MODEL`: constant model for var+const (default: `qwen-14b`)

## Reproducing Figures

You can either run the experiments above, or download pre-computed data from HuggingFace (link TBD) and place `results/` and `scaling/results/` in the project root. Then:

```bash
cd fig
python plot_figure_1.py           # Fig. 1: Tetris J vs B
python plot_figure_robustness.py  # Fig. 2: Prompt + reward robustness
python plot_figure_2.py           # Fig. 3: Cross-domain validation
python plot_figure_alpha.py       # Fig. 4: α(k) transition
python plot_figure_response.py    # Fig. 4: Coupling regimes (schematic, no data needed)
python plot_figure_3.py           # Fig. 5: Nested vs fixed AIME
python plot_appendix_figures.py   # Extended Data figures
```
