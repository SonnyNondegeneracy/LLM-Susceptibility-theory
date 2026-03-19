"""
Knapsack V2 Experiment: Le Chatelier's Principle

Harder knapsack instances for better signal:
- 50 items (up from 30)
- 30% capacity (down from 40%)
- 50 seeds (up from 10)
"""

import json
import os
import sys
import random
import concurrent.futures
from typing import List, Dict, Any

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from experiment_knapsack import (
    evaluate_packing, knapsack_beam_search,
    SYSTEM_PROMPTS_KNAPSACK, format_knapsack_for_llm, call_llm_knapsack,
)

with open(os.path.join(script_dir, 'api_key.json'), 'r') as f:
    API_KEYS = json.load(f)


# ==================== V2 Parameters ====================

N_ITEMS = 50
CAPACITY_RATIO = 0.30
SEEDS = list(range(42, 92))  # 50 seeds
BEAM_WIDTHS = [1, 2, 4, 8, 16, 32, 64]
TOP_K = 3
RESULTS_DIR = os.path.join(script_dir, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LLM_MODELS = {
    "qwen-7b": API_KEYS["qwen-7b"][0],
    "qwen-14b": API_KEYS["qwen-14b"][0],
    "qwen-32b": API_KEYS["qwen-32b"][0],
    "qwen-72b": API_KEYS["qwen-72b"][0],
    "qwen3-max": API_KEYS["qwen3_new"][0],
}


def generate_knapsack_v2(n_items: int, seed: int) -> Dict[str, Any]:
    """Generate harder knapsack instance: more items, tighter capacity."""
    rng = random.Random(seed)
    items = []
    for i in range(n_items):
        weight = rng.randint(1, 50)
        value = rng.randint(1, 100)
        items.append({'id': i, 'weight': weight, 'value': value,
                      'density': value / weight})
    total_weight = sum(it['weight'] for it in items)
    capacity = int(total_weight * CAPACITY_RATIO)
    return {'items': items, 'capacity': capacity, 'n_items': n_items}


# ==================== Runners ====================

def run_single_dfs(seed: int, beam_width: int) -> Dict[str, Any]:
    instance = generate_knapsack_v2(N_ITEMS, seed)
    packings = knapsack_beam_search(instance['items'], instance['capacity'],
                                     beam_width, top_k=1)
    best = packings[0] if packings else {'total_value': 0, 'total_weight': 0, 'selected': []}
    return {
        'agent_type': 'dfs',
        'model': None,
        'prompt': None,
        'seed': seed,
        'beam_width': beam_width,
        'total_value': best['total_value'],
        'total_weight': best['total_weight'],
        'n_selected': len(best['selected']),
        'capacity': instance['capacity'],
        'n_items': N_ITEMS,
        'capacity_ratio': CAPACITY_RATIO,
    }


def run_single_llm(seed: int, beam_width: int, model_key: str,
                    model_config: Dict[str, Any],
                    prompt_key: str = "standard") -> Dict[str, Any]:
    instance = generate_knapsack_v2(N_ITEMS, seed)
    packings = knapsack_beam_search(instance['items'], instance['capacity'],
                                     beam_width, top_k=TOP_K)
    if not packings:
        return {
            'agent_type': 'llm', 'model': model_key, 'prompt': prompt_key,
            'seed': seed, 'beam_width': beam_width,
            'total_value': 0, 'total_weight': 0, 'n_selected': 0,
            'capacity': instance['capacity'],
            'n_items': N_ITEMS, 'capacity_ratio': CAPACITY_RATIO,
        }

    system_prompt = SYSTEM_PROMPTS_KNAPSACK[prompt_key]
    user_msg = format_knapsack_for_llm(instance, packings)

    chosen = call_llm_knapsack(
        model_config['base_url'], model_config['api_key'],
        model_config['model'], system_prompt, user_msg,
        len(packings), timeout=15
    )

    best = packings[min(chosen, len(packings) - 1)]
    return {
        'agent_type': 'llm',
        'model': model_key,
        'prompt': prompt_key,
        'seed': seed,
        'beam_width': beam_width,
        'total_value': best['total_value'],
        'total_weight': best['total_weight'],
        'n_selected': len(best['selected']),
        'capacity': instance['capacity'],
        'n_items': N_ITEMS,
        'capacity_ratio': CAPACITY_RATIO,
    }


def run_dfs_baseline():
    print("  Knapsack V2 DFS baseline (50 items, 30% cap, 50 seeds)...")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(run_single_dfs, seed, bw)
                   for bw in BEAM_WIDTHS for seed in SEEDS]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"    DFS error: {e}")

    outpath = os.path.join(RESULTS_DIR, "knapsack_v2_dfs_baseline.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    Saved {len(results)} results to {outpath}")

    # Verify signal
    import numpy as np
    bw1 = [r['total_value'] for r in results if r['beam_width'] == 1]
    bw64 = [r['total_value'] for r in results if r['beam_width'] == 64]
    if bw1 and bw64:
        m1, m64 = np.mean(bw1), np.mean(bw64)
        s1, s64 = np.std(bw1, ddof=1), np.std(bw64, ddof=1)
        print(f"    BW=1:  {m1:.1f} +/- {s1:.1f}")
        print(f"    BW=64: {m64:.1f} +/- {s64:.1f}")
        print(f"    Gap:   {m64-m1:.1f}")
    return results


def run_llm_experiment(model_key: str, model_config: Dict[str, Any],
                        prompt_key: str = "standard"):
    print(f"  Knapsack V2 LLM: model={model_key}, prompt={prompt_key}")
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(run_single_llm, seed, bw, model_key,
                             model_config, prompt_key)
                   for bw in BEAM_WIDTHS for seed in SEEDS]
        done = 0
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
                done += 1
                if done % 50 == 0:
                    print(f"    {model_key}/{prompt_key}: {done}/{len(futures)}")
            except Exception as e:
                print(f"    LLM error: {e}")
                done += 1

    outpath = os.path.join(RESULTS_DIR, f"knapsack_v2_llm_{model_key}_{prompt_key}.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    Saved {len(results)} results to {outpath}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["dfs", "llm", "llm-prompts", "all"], default="all")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="standard")
    args = parser.parse_args()

    print("=" * 60)
    print("KNAPSACK V2 EXPERIMENT (50 items, 30% cap, 50 seeds)")
    print("=" * 60)

    if args.phase in ("dfs", "all"):
        run_dfs_baseline()

    if args.phase in ("llm", "all"):
        if args.model:
            run_llm_experiment(args.model, LLM_MODELS[args.model], args.prompt)
        else:
            for mk, mc in LLM_MODELS.items():
                run_llm_experiment(mk, mc, "standard")

    if args.phase in ("llm-prompts", "all"):
        for pk in SYSTEM_PROMPTS_KNAPSACK:
            if pk == "standard" and args.phase == "all":
                continue
            run_llm_experiment("qwen-32b", LLM_MODELS["qwen-32b"], pk)


if __name__ == "__main__":
    main()
