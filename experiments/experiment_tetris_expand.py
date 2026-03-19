"""
Expand Tetris experiments from 10 seeds to 40 seeds.
Adds seeds 52-81 to existing data (seeds 42-51).
Runs DFS (all reward functions) and all 5 LLM models (aggressive) in parallel.
"""
import json
import os
import sys
import time
import concurrent.futures

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from experiment import (
    run_dfs_game, run_llm_game,
    BEAM_WIDTHS, DFS_DEPTH, MAX_STEPS, GARBAGE_LINES,
    REWARD_FUNCTIONS_TO_TEST, LLM_MODELS, OUTPUT_DIR,
)

NEW_SEEDS = list(range(52, 82))  # 30 new seeds


def run_dfs_new():
    """Run DFS baseline for new seeds (fast, CPU-only)."""
    print(f"DFS: {len(NEW_SEEDS)} seeds × {len(BEAM_WIDTHS)} bw × {len(REWARD_FUNCTIONS_TO_TEST)} rf")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = []
        for rf in REWARD_FUNCTIONS_TO_TEST:
            for bw in BEAM_WIDTHS:
                for seed in NEW_SEEDS:
                    futures.append(executor.submit(
                        run_dfs_game, seed, bw, rf, DFS_DEPTH, MAX_STEPS, GARBAGE_LINES
                    ))
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    print(f"  DFS done: {len(results)} new records")
    return results


def run_llm_model(model_key, model_config):
    """Run one LLM model for new seeds with high parallelism."""
    print(f"  LLM {model_key}: starting {len(NEW_SEEDS)*len(BEAM_WIDTHS)} games...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        for bw in BEAM_WIDTHS:
            for seed in NEW_SEEDS:
                futures.append(executor.submit(
                    run_llm_game, seed, bw, "aggressive",
                    DFS_DEPTH, MAX_STEPS, GARBAGE_LINES,
                    model_key, model_config, top_k=3
                ))
        done = 0
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
                done += 1
                if done % 30 == 0:
                    print(f"    {model_key}: {done}/{len(futures)}")
            except Exception as e:
                print(f"    {model_key} failed: {e}")
                done += 1
    return model_key, results


def main():
    t0 = time.time()

    # Phase 1: DFS (fast)
    print("=" * 60)
    print("Phase 1: DFS baseline for new seeds")
    print("=" * 60)
    dfs_new = run_dfs_new()

    # Merge with existing
    dfs_path = os.path.join(OUTPUT_DIR, "dfs_baseline.json")
    with open(dfs_path) as f:
        dfs_old = json.load(f)
    dfs_all = dfs_old + dfs_new
    with open(dfs_path, 'w') as f:
        json.dump(dfs_all, f, indent=2)
    print(f"DFS merged: {len(dfs_old)} old + {len(dfs_new)} new = {len(dfs_all)} total")

    # Phase 2: All LLM models in parallel
    print("=" * 60)
    print("Phase 2: All 5 LLM models in parallel")
    print("=" * 60)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(run_llm_model, mk, mc): mk
            for mk, mc in LLM_MODELS.items()
        }
        for f in concurrent.futures.as_completed(futures):
            model_key, results = f.result()
            # Merge with existing
            llm_path = os.path.join(OUTPUT_DIR, f"llm_{model_key}_aggressive.json")
            with open(llm_path) as fj:
                old = json.load(fj)
            combined = old + results
            with open(llm_path, 'w') as fj:
                json.dump(combined, fj, indent=2)
            print(f"  {model_key}: {len(old)} old + {len(results)} new = {len(combined)} total")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
