"""
Run var+const for all selector models (7B, 32B, 72B, Qwen3).
14B already done in phase3_lechatelier.json.

Reuses existing generations and evaluate_config from lechatelier experiment.
Results appended to phase3_lechatelier.json.
"""
import json
import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

from hard_math_problems import generate_all_problems
from experiment_phase3 import K_VALUES
from experiment_phase3_lechatelier import (
    MODEL_ORDER, MODEL_LABELS, RESULTS_FILE, GENERATION_CHECKPOINT,
    load_generations, evaluate_config,
)

SELECTORS_TODO = ["qwen-7b", "qwen-32b", "qwen-72b", "qwen3-max"]


def main():
    print("=" * 60)
    print("var+const for all selector models")
    print("=" * 60)

    problems = generate_all_problems()
    print(f"  Loaded {len(problems)} problems")

    gen_index = load_generations()
    print(f"  Loaded generations for {len(gen_index)} (model, problem) pairs")

    # Load existing results
    existing = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
        print(f"  Loaded {len(existing)} existing results")

    all_new = []
    for sel_model in SELECTORS_TODO:
        print(f"\n  === var+const (selector={sel_model}) ===")
        t0 = time.time()
        for k in K_VALUES:
            for gen_model in MODEL_ORDER:
                print(f"    k={k:2d} gen={gen_model:>12s} sel={sel_model}...", end=" ", flush=True)
                res = evaluate_config(gen_index, problems, gen_model, sel_model, k)
                for r in res:
                    r["config"] = "var+const"
                all_new.extend(res)
                ag = [r for r in res if r["method"] == "agent"]
                mv = [r for r in res if r["method"] == "majority_vote"]
                ag_acc = sum(r["correct"] for r in ag) / len(ag) * 100 if ag else 0
                mv_acc = sum(r["correct"] for r in mv) / len(mv) * 100 if mv else 0
                print(f"MV={mv_acc:.1f}% Agent={ag_acc:.1f}%")
        elapsed = time.time() - t0
        print(f"  Done {sel_model} in {elapsed:.0f}s")

    # Append to existing results
    combined = existing + all_new
    with open(RESULTS_FILE, 'w') as f:
        json.dump(combined, f, indent=1)
    print(f"\n  Saved {len(combined)} total results to {RESULTS_FILE}")
    print(f"  ({len(existing)} existing + {len(all_new)} new)")


if __name__ == "__main__":
    main()
