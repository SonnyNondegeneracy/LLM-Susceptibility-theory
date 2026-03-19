"""
Phase 3 Le Chatelier Experiment: Disentangling Generator vs Selector

Phase 3 (var+var) used the same model for generation and selection.
This experiment separates them:

  var+const: generator varies [7b..qwen3], selector fixed at CONST_MODEL
  const+var: generator fixed at CONST_MODEL, selector varies [7b..qwen3]

Reuses existing generations from phase3_generations.json — only new agent
selection calls needed (~2,700 per configuration).
"""

import json
import os
import sys
import re
import time
import argparse
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

from hard_math_problems import generate_all_problems
from experiment_phase3 import (
    LLM_MODELS, call_llm_generic, parse_number, parse_selection,
    answers_match, majority_vote, deduplicate_answers,
    AGENT_SELECT_SYSTEM, AGENT_SELECT_USER,
    K_VALUES, K_MAX, MAX_WORKERS,
)

RESULTS_DIR = os.path.join(script_dir, "results")
PLOTS_DIR = os.path.join(script_dir, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_ORDER = ["qwen-7b", "qwen-14b", "qwen-32b", "qwen-72b", "qwen3-max"]
MODEL_LABELS = ["7B", "14B", "32B", "72B", "Qwen3"]
MODEL_PARAMS = np.array([7e9, 14e9, 32e9, 72e9, 200e9])
LOG_PARAMS = np.log10(MODEL_PARAMS)

GENERATION_CHECKPOINT = os.path.join(RESULTS_DIR, "phase3_generations.json")
RESULTS_FILE = os.path.join(RESULTS_DIR, "phase3_lechatelier.json")


def load_generations():
    with open(GENERATION_CHECKPOINT, 'r') as f:
        gens = json.load(f)
    # Index by (model, problem_id) -> sorted list
    index = {}
    for g in gens:
        key = (g["model"], g["problem_id"])
        if key not in index:
            index[key] = []
        index[key].append(g)
    for key in index:
        index[key].sort(key=lambda x: x["attempt"])
    return index


def agent_select(selector_model: str, problem: Dict, unique_answers: List[float]) -> Optional[float]:
    """LLM selects from deduplicated options using a specific selector model."""
    if len(unique_answers) == 1:
        return unique_answers[0]

    options_text = "\n".join(f"  Option {i+1}: {int(a) if a == int(a) else a}"
                            for i, a in enumerate(unique_answers))
    cfg = LLM_MODELS[selector_model]
    prompt = AGENT_SELECT_USER.format(
        question=problem["question"],
        options_text=options_text,
    )
    raw = call_llm_generic(
        cfg["base_url"], cfg["api_key"], cfg["model"],
        AGENT_SELECT_SYSTEM, prompt, temperature=0.1
    )
    idx = parse_selection(raw, len(unique_answers))
    if idx is not None:
        return unique_answers[idx - 1]
    parsed = parse_number(raw)
    if parsed is not None:
        for u in unique_answers:
            if answers_match(parsed, u):
                return u
    return None


def evaluate_config(gen_index, problems, generator_model, selector_model, k):
    """Evaluate a specific generator+selector configuration at given k."""
    prob_by_id = {p["id"]: p for p in problems}
    results = []

    # Collect tasks
    tasks = []
    for prob in problems:
        key = (generator_model, prob["id"])
        gens = gen_index.get(key, [])[:k]
        answers = [g["parsed_answer"] for g in gens if g["parsed_answer"] is not None]
        unique = deduplicate_answers(answers) if answers else []

        # Majority vote (no LLM call)
        mv_answer = majority_vote(answers) if answers else None
        mv_correct = mv_answer is not None and answers_match(mv_answer, prob["answer"])
        results.append({
            "method": "majority_vote",
            "generator": generator_model,
            "selector": "N/A",
            "model": generator_model,
            "problem_id": prob["id"],
            "problem_type": prob["type"],
            "k": k,
            "answer": mv_answer,
            "true_answer": prob["answer"],
            "correct": mv_correct,
            "n_unique": len(unique),
        })

        # Agent selection task
        tasks.append((prob, answers, unique))

    # Run agent selections in parallel
    agent_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for prob, answers, unique in tasks:
            if not unique:
                agent_map[prob["id"]] = None
            elif len(unique) == 1:
                agent_map[prob["id"]] = unique[0]
            else:
                f = ex.submit(agent_select, selector_model, prob, unique)
                futures[f] = prob["id"]

        for f in concurrent.futures.as_completed(futures):
            try:
                agent_map[futures[f]] = f.result()
            except Exception:
                agent_map[futures[f]] = None

    for prob, answers, unique in tasks:
        ag_answer = agent_map.get(prob["id"])
        ag_correct = ag_answer is not None and answers_match(ag_answer, prob["answer"])
        results.append({
            "method": "agent",
            "generator": generator_model,
            "selector": selector_model,
            "model": f"{generator_model}+{selector_model}",
            "problem_id": prob["id"],
            "problem_type": prob["type"],
            "k": k,
            "answer": ag_answer,
            "true_answer": prob["answer"],
            "correct": ag_correct,
            "n_unique": len(unique),
        })

    return results


def run_experiment(gen_index, problems, const_model="qwen-14b"):
    """Run var+const and const+var experiments."""
    all_results = []

    # var+const: generator varies, selector = const_model
    print(f"\n  === var+const (selector={const_model}) ===")
    for k in K_VALUES:
        for gen_model in MODEL_ORDER:
            print(f"    k={k:2d} gen={gen_model:>10s} sel={const_model}...", end=" ", flush=True)
            res = evaluate_config(gen_index, problems, gen_model, const_model, k)
            for r in res:
                r["config"] = "var+const"
            all_results.extend(res)
            ag = [r for r in res if r["method"] == "agent"]
            mv = [r for r in res if r["method"] == "majority_vote"]
            ag_acc = sum(r["correct"] for r in ag) / len(ag) * 100 if ag else 0
            mv_acc = sum(r["correct"] for r in mv) / len(mv) * 100 if mv else 0
            print(f"MV={mv_acc:.1f}% Agent={ag_acc:.1f}%")

    # const+var: generator = const_model, selector varies
    print(f"\n  === const+var (generator={const_model}) ===")
    for k in K_VALUES:
        for sel_model in MODEL_ORDER:
            print(f"    k={k:2d} gen={const_model:>10s} sel={sel_model}...", end=" ", flush=True)
            res = evaluate_config(gen_index, problems, const_model, sel_model, k)
            for r in res:
                r["config"] = "const+var"
            all_results.extend(res)
            ag = [r for r in res if r["method"] == "agent"]
            mv = [r for r in res if r["method"] == "majority_vote"]
            ag_acc = sum(r["correct"] for r in ag) / len(ag) * 100 if ag else 0
            mv_acc = sum(r["correct"] for r in mv) / len(mv) * 100 if mv else 0
            print(f"MV={mv_acc:.1f}% Agent={ag_acc:.1f}%")

    return all_results


# ==================== Analysis & Plots ====================

COLORS = {'agent': '#2196F3', 'majority_vote': '#FF5722'}
MARKERS = {'agent': 's', 'majority_vote': 'o'}

def compute_scaling_slope(log_x, y_values):
    x = np.array(log_x[:len(y_values)])
    y = np.array(y_values)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    slope, _ = np.polyfit(x[mask], y[mask], 1)
    return slope


def plot_lechatelier(results, const_model):
    """Generate the Le Chatelier comparison plot."""

    # Load var+var results for comparison
    varvar_path = os.path.join(RESULTS_DIR, "phase3_results.json")
    varvar = []
    if os.path.exists(varvar_path):
        with open(varvar_path) as f:
            varvar = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    k_plot = 21  # Focus on k=21

    # --- Row 1: Accuracy vs model at k=21 for each config ---

    # Panel (0,0): var+var (from phase3_results.json)
    ax = axes[0][0]
    for method, label in [("agent", "Agent (var+var)"), ("majority_vote", "MajVote")]:
        accs = []
        for m in MODEL_ORDER:
            subset = [r for r in varvar if r["method"] == method and r["model"] == m and r["k"] == k_plot]
            accs.append(sum(r["correct"] for r in subset) / len(subset) if subset else np.nan)
        ax.plot(range(len(MODEL_ORDER)), accs, marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#333'), linewidth=2, markersize=8, label=label)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"var+var (k={k_plot})\nGen=X, Sel=X", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 0.35)

    # Panel (0,1): var+const
    ax = axes[0][1]
    vc_results = [r for r in results if r["config"] == "var+const"]
    for method, label in [("agent", f"Agent (sel={const_model})"), ("majority_vote", "MajVote")]:
        accs = []
        for m in MODEL_ORDER:
            subset = [r for r in vc_results if r["method"] == method
                      and r["generator"] == m and r["k"] == k_plot]
            accs.append(sum(r["correct"] for r in subset) / len(subset) if subset else np.nan)
        ax.plot(range(len(MODEL_ORDER)), accs, marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#333'), linewidth=2, markersize=8, label=label)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_title(f"var+const (k={k_plot})\nGen=X, Sel={const_model}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Generator Model", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 0.35)

    # Panel (0,2): const+var
    ax = axes[0][2]
    cv_results = [r for r in results if r["config"] == "const+var"]
    for method, label in [("agent", f"Agent (gen={const_model})"), ("majority_vote", f"MajVote (gen={const_model})")]:
        accs = []
        for m in MODEL_ORDER:
            subset = [r for r in cv_results if r["method"] == method
                      and (r["selector"] == m if method == "agent" else r["generator"] == const_model)
                      and r["k"] == k_plot]
            # For MajVote in const+var, it's same for all selector models (only depends on generator)
            if method == "majority_vote":
                subset = [r for r in cv_results if r["method"] == method
                          and r["generator"] == const_model and r["k"] == k_plot]
                accs.append(sum(r["correct"] for r in subset) / len(subset) if subset else np.nan)
            else:
                accs.append(sum(r["correct"] for r in subset) / len(subset) if subset else np.nan)
        if method == "majority_vote":
            # MajVote is constant line (doesn't depend on selector)
            mv_val = accs[0] if accs else np.nan
            ax.axhline(y=mv_val, color=COLORS["majority_vote"], linewidth=2,
                       linestyle='--', label=f"MajVote ({mv_val:.1%})")
        else:
            ax.plot(range(len(MODEL_ORDER)), accs, marker='s',
                    color=COLORS["agent"], linewidth=2, markersize=8, label=label)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_title(f"const+var (k={k_plot})\nGen={const_model}, Sel=X", fontsize=11, fontweight='bold')
    ax.set_xlabel("Selector Model", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 0.35)

    # --- Row 2: Accuracy vs k for each config ---

    # Panel (1,0): var+var avg across models
    ax = axes[1][0]
    for method, label in [("agent", "Agent"), ("majority_vote", "MajVote")]:
        avg_accs = []
        for k in K_VALUES:
            model_accs = []
            for m in MODEL_ORDER:
                subset = [r for r in varvar if r["method"] == method and r["model"] == m and r["k"] == k]
                if subset:
                    model_accs.append(sum(r["correct"] for r in subset) / len(subset))
            avg_accs.append(np.mean(model_accs) if model_accs else np.nan)
        ax.plot(K_VALUES, avg_accs, marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#333'), linewidth=2, markersize=8, label=label)
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Avg Accuracy", fontsize=11)
    ax.set_title("var+var: Avg Acc vs k", fontsize=11, fontweight='bold')
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (1,1): var+const avg across generator models
    ax = axes[1][1]
    for method, label in [("agent", f"Agent (sel={const_model})"), ("majority_vote", "MajVote")]:
        avg_accs = []
        for k in K_VALUES:
            model_accs = []
            for m in MODEL_ORDER:
                subset = [r for r in vc_results if r["method"] == method
                          and r["generator"] == m and r["k"] == k]
                if subset:
                    model_accs.append(sum(r["correct"] for r in subset) / len(subset))
            avg_accs.append(np.mean(model_accs) if model_accs else np.nan)
        ax.plot(K_VALUES, avg_accs, marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#333'), linewidth=2, markersize=8, label=label)
    ax.set_xlabel("k", fontsize=11)
    ax.set_title(f"var+const: Avg Acc vs k", fontsize=11, fontweight='bold')
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (1,2): const+var accuracy vs selector model at various k
    ax = axes[1][2]
    for k in [3, 9, 21]:
        accs = []
        for m in MODEL_ORDER:
            subset = [r for r in cv_results if r["method"] == "agent"
                      and r["selector"] == m and r["k"] == k]
            accs.append(sum(r["correct"] for r in subset) / len(subset) if subset else np.nan)
        ax.plot(range(len(MODEL_ORDER)), accs, marker='s', linewidth=2, markersize=6,
                label=f"Agent k={k}")
    # MajVote baseline at k=21
    mv_subset = [r for r in cv_results if r["method"] == "majority_vote"
                 and r["generator"] == const_model and r["k"] == 21]
    if mv_subset:
        mv_val = sum(r["correct"] for r in mv_subset) / len(mv_subset)
        ax.axhline(y=mv_val, color=COLORS["majority_vote"], linewidth=2,
                   linestyle='--', label=f"MajVote k=21 ({mv_val:.1%})")
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_xlabel("Selector Model", fontsize=10)
    ax.set_title(f"const+var: Agent by Selector\n(gen={const_model})", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 0.35)

    fig.suptitle(f"Le Chatelier: Disentangling Generator vs Selector (const={const_model})",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, "phase3_lechatelier.png")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {outpath}")


def print_summary(results, const_model):
    """Print summary table."""
    k = 21
    print(f"\n  {'='*70}")
    print(f"  SUMMARY at k={k} (const_model={const_model})")
    print(f"  {'='*70}")

    print(f"\n  var+const (generator varies, selector={const_model}):")
    print(f"  {'Generator':>10s}  {'MajVote':>8s}  {'Agent':>8s}  {'Δ':>8s}")
    vc = [r for r in results if r["config"] == "var+const" and r["k"] == k]
    for m in MODEL_ORDER:
        mv = [r for r in vc if r["method"] == "majority_vote" and r["generator"] == m]
        ag = [r for r in vc if r["method"] == "agent" and r["generator"] == m]
        mv_acc = sum(r["correct"] for r in mv) / len(mv) if mv else 0
        ag_acc = sum(r["correct"] for r in ag) / len(ag) if ag else 0
        print(f"  {m:>10s}  {mv_acc:>7.1%}  {ag_acc:>7.1%}  {ag_acc-mv_acc:>+7.1%}")

    print(f"\n  const+var (generator={const_model}, selector varies):")
    print(f"  {'Selector':>10s}  {'MajVote':>8s}  {'Agent':>8s}  {'Δ':>8s}")
    cv = [r for r in results if r["config"] == "const+var" and r["k"] == k]
    # MajVote is same for all selectors
    mv_all = [r for r in cv if r["method"] == "majority_vote"]
    mv_acc = sum(r["correct"] for r in mv_all) / len(mv_all) if mv_all else 0
    for m in MODEL_ORDER:
        ag = [r for r in cv if r["method"] == "agent" and r["selector"] == m]
        ag_acc = sum(r["correct"] for r in ag) / len(ag) if ag else 0
        print(f"  {m:>10s}  {mv_acc:>7.1%}  {ag_acc:>7.1%}  {ag_acc-mv_acc:>+7.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--const", default="qwen-14b", help="Constant model for var+const / const+var")
    args = parser.parse_args()

    const_model = args.const

    print("=" * 60)
    print(f"PHASE 3 LE CHATELIER: var+const / const+var")
    print(f"Constant model: {const_model}")
    print("=" * 60)

    problems = generate_all_problems()
    print(f"  Loaded {len(problems)} problems")

    print("  Loading generations...")
    gen_index = load_generations()
    print(f"  Loaded generations for {len(gen_index)} (model, problem) pairs")

    t0 = time.time()
    results = run_experiment(gen_index, problems, const_model)
    elapsed = time.time() - t0
    print(f"\n  Experiment complete in {elapsed:.0f}s")

    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"  Saved {RESULTS_FILE}")

    # Summary
    print_summary(results, const_model)

    # Plots
    print("\n  Generating plots...")
    plot_lechatelier(results, const_model)

    print("\n" + "=" * 60)
    print("LE CHATELIER EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
