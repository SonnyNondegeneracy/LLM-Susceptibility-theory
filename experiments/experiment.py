"""
Formal Experiment: Le Chatelier's Principle for AI Skill Management

Research Question:
    When using an LLM to manage/select from algorithmic search results (beam search),
    does the performance improvement from increased compute budget (beam width)
    scale slower than or equal to pure algorithmic search?

Design:
    - Independent Variable: beam_width (proxy for compute budget)
    - Conditions:
        1. Pure beam search (DFS) - directly uses best action
        2. LLM-managed beam search - LLM selects from top-k suggestions
    - Multiple LLM models tested for generality
    - Multiple seeds for statistical significance
    - Multiple reward functions for robustness
"""

import json
import os
import sys
import time
import random
import traceback
import concurrent.futures
from typing import Dict, Any, List

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from game_engine import (
    GameState, REWARD_FUNCTIONS, default_reward, aggressive_reward,
    conservative_reward, dfs_search, get_possible_placements
)
from llm_agent import LLMAgent, llm_play_game

# ==================== Configuration ====================

# Load API keys
with open(os.path.join(script_dir, 'api_key.json'), 'r') as f:
    API_KEYS = json.load(f)

# Experiment parameters
BEAM_WIDTHS = [1, 2, 4, 8, 16, 32]
SEEDS = list(range(42, 52))  # 10 seeds (sufficient for statistical significance)
DFS_DEPTH = 3
MAX_STEPS = 50  # Short games to keep LLM calls manageable (50 steps * 10 seeds * 6 bw = 3000 calls/model)
GARBAGE_LINES = 6
REWARD_FUNCTIONS_TO_TEST = ["aggressive", "default", "conservative"]

# LLM models to test (from smallest to largest)
LLM_MODELS = {
    "qwen-7b": API_KEYS["qwen-7b"][0],
    "qwen-14b": API_KEYS["qwen-14b"][0],
    "qwen-32b": API_KEYS["qwen-32b"][0],
    "qwen-72b": API_KEYS["qwen-72b"][0],
    "qwen3-max": API_KEYS["qwen3_new"][0],
}

OUTPUT_DIR = os.path.join(script_dir, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== Experiment Functions ====================

def run_dfs_game(seed: int, beam_width: int, reward_fn_name: str,
                 depth: int, max_steps: int, garbage_lines: int) -> Dict[str, Any]:
    """Run a single game with pure beam search (no LLM)."""
    game = GameState(seed=seed)
    game.add_garbage_lines(garbage_lines)

    reward_fn = REWARD_FUNCTIONS[reward_fn_name]
    steps = 0

    while not game.game_over and steps < max_steps:
        if game.current_piece is None:
            break
        result = dfs_search(game, reward_fn, depth, 1, beam_width)
        if not result.top_k or not result.top_k[0][1]:
            break
        target_col = result.top_k[0][1][0]['target_col']
        game.drop_and_place(target_col)
        steps += 1

    return {
        'agent_type': 'dfs',
        'model': None,
        'seed': seed,
        'beam_width': beam_width,
        'reward_fn': reward_fn_name,
        'depth': depth,
        'score': game.score,
        'lines_cleared': game.lines_cleared,
        'steps_taken': steps,
        'game_over': game.game_over,
    }


def run_llm_game(seed: int, beam_width: int, reward_fn_name: str,
                 depth: int, max_steps: int, garbage_lines: int,
                 model_key: str, model_config: Dict[str, Any],
                 top_k: int = 3) -> Dict[str, Any]:
    """Run a single game with LLM-managed beam search."""
    game = GameState(seed=seed)
    game.add_garbage_lines(garbage_lines)

    agent = LLMAgent(
        base_url=model_config['base_url'],
        api_key=model_config['api_key'],
        model_name=model_config['model'],
        reward_function=reward_fn_name,
        dfs_depth=depth,
        dfs_width=beam_width,
        dfs_top_k=top_k,
        use_dfs_hint=True,
        temperature=0.1,
        max_retries=2,
        retry_delay=0.5,
        timeout=15,
    )

    result = llm_play_game(game, agent, max_steps=max_steps)

    return {
        'agent_type': 'llm',
        'model': model_key,
        'seed': seed,
        'beam_width': beam_width,
        'reward_fn': reward_fn_name,
        'depth': depth,
        'top_k': top_k,
        'score': game.score,
        'lines_cleared': game.lines_cleared,
        'steps_taken': result['steps_taken'],
        'game_over': game.game_over,
        'error_stats': result['error_stats'],
    }


# ==================== Phase 1: DFS Baseline ====================

def run_dfs_baseline():
    """Run all DFS baseline experiments (fast, no API calls)."""
    print("=" * 60)
    print("PHASE 1: Running DFS Baseline")
    print("=" * 60)

    results = []
    total = len(BEAM_WIDTHS) * len(SEEDS) * len(REWARD_FUNCTIONS_TO_TEST)
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = []
        for reward_fn_name in REWARD_FUNCTIONS_TO_TEST:
            for beam_width in BEAM_WIDTHS:
                for seed in SEEDS:
                    futures.append(executor.submit(
                        run_dfs_game, seed, beam_width, reward_fn_name,
                        DFS_DEPTH, MAX_STEPS, GARBAGE_LINES
                    ))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                done += 1
                if done % 50 == 0:
                    print(f"  DFS progress: {done}/{total}")
            except Exception as e:
                print(f"  DFS job failed: {e}")
                done += 1

    # Save results
    outpath = os.path.join(OUTPUT_DIR, "dfs_baseline.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"DFS baseline saved: {outpath} ({len(results)} results)")
    return results


# ==================== Phase 2: LLM Experiments ====================

def run_llm_experiment(model_key: str, model_config: Dict[str, Any],
                       reward_fn_name: str = "aggressive"):
    """Run LLM experiment for one model and one reward function."""
    print(f"\n  Running LLM experiment: model={model_key}, reward={reward_fn_name}")

    results = []
    total = len(BEAM_WIDTHS) * len(SEEDS)
    done = 0

    # Use more threads for parallel API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for beam_width in BEAM_WIDTHS:
            for seed in SEEDS:
                futures.append(executor.submit(
                    run_llm_game, seed, beam_width, reward_fn_name,
                    DFS_DEPTH, MAX_STEPS, GARBAGE_LINES,
                    model_key, model_config, top_k=3
                ))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                done += 1
                if done % 10 == 0:
                    print(f"    {model_key}/{reward_fn_name} progress: {done}/{total}")
            except Exception as e:
                print(f"    {model_key} job failed: {e}")
                traceback.print_exc()
                done += 1

    return results


def run_all_llm_experiments():
    """Run LLM experiments for all models."""
    print("=" * 60)
    print("PHASE 2: Running LLM Experiments")
    print("=" * 60)

    all_results = []

    # Primary experiment: all models with aggressive reward function
    for model_key, model_config in LLM_MODELS.items():
        results = run_llm_experiment(model_key, model_config, "aggressive")
        all_results.extend(results)

        # Save intermediate results
        outpath = os.path.join(OUTPUT_DIR, f"llm_{model_key}_aggressive.json")
        with open(outpath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {outpath} ({len(results)} results)")

    # Robustness check: test one model (qwen-32b) with all reward functions
    robustness_model = "qwen-32b"
    robustness_config = LLM_MODELS[robustness_model]
    for reward_fn_name in ["default", "conservative"]:
        results = run_llm_experiment(robustness_model, robustness_config, reward_fn_name)
        all_results.extend(results)

        outpath = os.path.join(OUTPUT_DIR, f"llm_{robustness_model}_{reward_fn_name}.json")
        with open(outpath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {outpath} ({len(results)} results)")

    # Save all LLM results
    outpath = os.path.join(OUTPUT_DIR, "llm_all.json")
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll LLM results saved: {outpath} ({len(all_results)} results)")
    return all_results


# ==================== Main ====================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Le Chatelier Experiment")
    parser.add_argument("--phase", choices=["dfs", "llm", "all"], default="all",
                        help="Which phase to run")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this LLM model (e.g., qwen-7b)")
    parser.add_argument("--reward", type=str, default="aggressive",
                        help="Reward function for single-model LLM run")
    args = parser.parse_args()

    start_time = time.time()

    if args.phase in ("dfs", "all"):
        run_dfs_baseline()

    if args.phase in ("llm", "all"):
        if args.model:
            if args.model not in LLM_MODELS:
                print(f"Unknown model: {args.model}. Available: {list(LLM_MODELS.keys())}")
                sys.exit(1)
            results = run_llm_experiment(args.model, LLM_MODELS[args.model], args.reward)
            outpath = os.path.join(OUTPUT_DIR, f"llm_{args.model}_{args.reward}.json")
            with open(outpath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved: {outpath} ({len(results)} results)")
        else:
            run_all_llm_experiments()

    elapsed = time.time() - start_time
    print(f"\nTotal experiment time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
