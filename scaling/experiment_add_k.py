"""
Add k=17 and k=19 evaluation to existing phase3 data.
Reuses existing generations (k_max=21). Only new agent selection calls needed.
Appends results to phase3_results.json and phase3_lechatelier.json.
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
from experiment_phase3 import evaluate_at_k, MODEL_ORDER
from experiment_phase3_lechatelier import (
    load_generations, evaluate_config, RESULTS_FILE as LC_RESULTS_FILE,
)

RESULTS_DIR = os.path.join(script_dir, "results")
PHASE3_RESULTS = os.path.join(RESULTS_DIR, "phase3_results.json")
GENERATION_CHECKPOINT = os.path.join(RESULTS_DIR, "phase3_generations.json")

NEW_K = [17, 19]


def main():
    print("=" * 60)
    print(f"Adding k={NEW_K} to phase3 experiments")
    print("=" * 60)

    problems = generate_all_problems()
    print(f"  Loaded {len(problems)} problems")

    # Load generations
    with open(GENERATION_CHECKPOINT, 'r') as f:
        generations = json.load(f)
    print(f"  Loaded {len(generations)} generations")

    gen_index = load_generations()
    print(f"  Indexed {len(gen_index)} (model, problem) pairs")

    # === Phase 1: var+var (phase3_results.json) ===
    print("\n" + "=" * 60)
    print("Phase 1: var+var at new k values")
    print("=" * 60)

    with open(PHASE3_RESULTS) as f:
        existing_results = json.load(f)
    print(f"  Existing: {len(existing_results)} records")

    new_varvar = []
    for k in NEW_K:
        print(f"\n  Evaluating k={k}...")
        t0 = time.time()
        results_k = evaluate_at_k(generations, problems, k)
        new_varvar.extend(results_k)
        elapsed = time.time() - t0

        for method in ["majority_vote", "agent"]:
            for model_key in MODEL_ORDER:
                subset = [r for r in results_k
                          if r["method"] == method and r["model"] == model_key]
                acc = sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0
                print(f"    k={k:2d} {method:>14s} {model_key:>10s}: {acc:.1f}%")
        print(f"    Done in {elapsed:.0f}s")

    combined = existing_results + new_varvar
    with open(PHASE3_RESULTS, 'w') as f:
        json.dump(combined, f, indent=1)
    print(f"\n  Saved {len(combined)} total to {PHASE3_RESULTS}")
    print(f"  ({len(existing_results)} existing + {len(new_varvar)} new)")

    # === Phase 2: var+const for all 5 selectors (phase3_lechatelier.json) ===
    print("\n" + "=" * 60)
    print("Phase 2: var+const at new k values (all 5 selectors)")
    print("=" * 60)

    with open(LC_RESULTS_FILE) as f:
        existing_lc = json.load(f)
    print(f"  Existing: {len(existing_lc)} records")

    new_lc = []
    for sel_model in MODEL_ORDER:
        print(f"\n  === selector={sel_model} ===")
        t0 = time.time()
        for k in NEW_K:
            for gen_model in MODEL_ORDER:
                print(f"    k={k:2d} gen={gen_model:>12s} sel={sel_model}...", end=" ", flush=True)
                res = evaluate_config(gen_index, problems, gen_model, sel_model, k)
                for r in res:
                    r["config"] = "var+const"
                new_lc.extend(res)
                ag = [r for r in res if r["method"] == "agent"]
                mv = [r for r in res if r["method"] == "majority_vote"]
                ag_acc = sum(r["correct"] for r in ag) / len(ag) * 100 if ag else 0
                mv_acc = sum(r["correct"] for r in mv) / len(mv) * 100 if mv else 0
                print(f"MV={mv_acc:.1f}% Agent={ag_acc:.1f}%")
        elapsed = time.time() - t0
        print(f"  Done {sel_model} in {elapsed:.0f}s")

    combined_lc = existing_lc + new_lc
    with open(LC_RESULTS_FILE, 'w') as f:
        json.dump(combined_lc, f, indent=1)
    print(f"\n  Saved {len(combined_lc)} total to {LC_RESULTS_FILE}")
    print(f"  ({len(existing_lc)} existing + {len(new_lc)} new)")

    print("\nDone!")


if __name__ == "__main__":
    main()
