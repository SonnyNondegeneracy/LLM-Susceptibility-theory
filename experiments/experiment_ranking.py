"""
Ranking Experiment: LLM Beats Algorithm at Low Budget

Noisy ranking with world knowledge — demonstrates the OTHER side of Le Chatelier:
- Algorithm generates candidates with noisy scores (can't determine the true best)
- LLM has world knowledge to judge which candidate is truly best
- At low budget: algorithm's ranking is noisy → LLM's knowledge wins
- At high budget: algorithm converges to truth → LLM advantage vanishes

This shows LLM adds value when algorithm is weak, but value diminishes as algorithm improves.
"""

import json
import os
import sys
import math
import random
import concurrent.futures
from typing import List, Tuple, Dict, Any, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

with open(os.path.join(script_dir, 'api_key.json'), 'r') as f:
    API_KEYS = json.load(f)


# ==================== Datasets with True Values ====================

DATASETS = {
    "gdp": {
        "name": "Countries by GDP (Trillion USD)",
        "metric": "GDP",
        "unit": "trillion USD",
        "items": [
            ("USA", 25.5), ("China", 17.9), ("Japan", 4.2), ("Germany", 4.1),
            ("India", 3.4), ("UK", 3.1), ("France", 2.8), ("Italy", 2.0),
            ("Canada", 2.0), ("Brazil", 1.9), ("Russia", 1.8), ("South Korea", 1.7),
            ("Australia", 1.7), ("Mexico", 1.3), ("Spain", 1.4),
        ],
        "sigma": 8.0,  # USA-China gap=7.6, P(correct)~50% at budget=1
    },
    "population": {
        "name": "Countries by Population (Millions)",
        "metric": "population",
        "unit": "million people",
        "items": [
            ("India", 1428), ("China", 1412), ("USA", 332), ("Indonesia", 275),
            ("Pakistan", 225), ("Brazil", 214), ("Nigeria", 213), ("Bangladesh", 170),
            ("Russia", 144), ("Mexico", 128), ("Japan", 125), ("Ethiopia", 120),
            ("Philippines", 113), ("Egypt", 104), ("DR Congo", 99),
        ],
        "sigma": 50.0,  # India-China gap=16, very close
    },
    "planets": {
        "name": "Planets by Diameter (km)",
        "metric": "diameter",
        "unit": "km",
        "items": [
            ("Jupiter", 139820), ("Saturn", 116460), ("Uranus", 50724),
            ("Neptune", 49244), ("Earth", 12742), ("Venus", 12104),
            ("Mars", 6779), ("Mercury", 4879),
        ],
        "sigma": 30000.0,  # Jupiter-Saturn gap=23360, higher sigma for more noise
    },
    "animals": {
        "name": "Animals by Weight (kg)",
        "metric": "weight",
        "unit": "kg",
        "items": [
            ("African Elephant", 6000), ("White Rhino", 2300), ("Hippo", 1500),
            ("Giraffe", 1200), ("Horse", 500), ("Polar Bear", 450),
            ("Lion", 190), ("Gorilla", 160), ("Human", 70), ("Dog", 30),
            ("Cat", 5), ("Rabbit", 2),
        ],
        "sigma": 4000.0,  # Elephant-Rhino gap=3700, want ~50% at budget=1
    },
}


# ==================== Algorithm: Noisy Ranking ====================

def noisy_rank_algorithm(dataset: Dict, budget: int, seed: int,
                          top_k: int = 5) -> Dict[str, Any]:
    """
    Algorithm estimates each item's score = true_score + Gaussian_noise(0, sigma/sqrt(budget)).
    Returns top-K candidates sorted by estimated score (descending).
    """
    rng = random.Random(seed)
    items = dataset["items"]
    sigma = dataset["sigma"]
    noise_scale = sigma / math.sqrt(budget)

    # Generate noisy estimates
    estimates = []
    for name, true_val in items:
        noise = rng.gauss(0, noise_scale)
        estimated = true_val + noise
        estimates.append((name, true_val, estimated))

    # Sort by estimated score (descending)
    estimates.sort(key=lambda x: x[2], reverse=True)

    # The algorithm picks the one with highest estimated score
    algo_pick = estimates[0][0]
    true_best = max(items, key=lambda x: x[1])[0]
    algo_correct = (algo_pick == true_best)

    # Return top-K for LLM
    top_k_candidates = estimates[:top_k]

    return {
        "algo_pick": algo_pick,
        "algo_correct": algo_correct,
        "true_best": true_best,
        "top_k": [(name, true_val, est) for name, true_val, est in top_k_candidates],
        "noise_scale": noise_scale,
    }


# ==================== LLM Agent for Ranking ====================

SYSTEM_PROMPTS_RANKING = {
    "minimal": """Pick the item that truly ranks #1 by the given metric.
Reply ONLY with JSON: {"item_index": <0-based index from the list>}""",

    "standard": """You are identifying which item truly ranks #1 by a given real-world metric.
An algorithm estimated scores with measurement noise. The estimates may be inaccurate.
Review the candidates and pick the one you believe is truly #1.

RESPONSE FORMAT:
```json
{"thinking": "your analysis", "item_index": <0-based index from the list>}
```""",

    "knowledge": """You are identifying which item truly ranks #1 by a given real-world metric.
An algorithm estimated scores with measurement noise — the estimates are UNRELIABLE at low sample sizes.

IMPORTANT: Use your real-world knowledge! You likely know the true ranking of these items.
If the noisy estimates disagree with what you know to be true, TRUST YOUR KNOWLEDGE over the numbers.

RESPONSE FORMAT:
```json
{"thinking": "what I know about these items", "item_index": <0-based index from the list>}
```""",
}


def format_ranking_for_llm(dataset: Dict, algo_result: Dict) -> str:
    """Format the ranking task for the LLM."""
    metric = dataset["metric"]
    unit = dataset["unit"]
    top_k = algo_result["top_k"]

    lines = [
        f"Which item truly has the HIGHEST {metric}?",
        f"The algorithm estimated {metric} with noise. Estimates may be inaccurate.",
        "",
        f"Algorithm's top-{len(top_k)} candidates (by estimated {metric}):",
    ]

    for idx, (name, true_val, est_val) in enumerate(top_k):
        lines.append(f"  {idx}. {name}: estimated {metric} = {est_val:.1f} {unit}")

    lines.append("")
    lines.append(f"Which item truly ranks #1 in {metric}? Give the 0-based index.")

    return "\n".join(lines)


def call_llm_ranking(base_url: str, api_key: str, model_name: str,
                      system_prompt: str, user_message: str,
                      n_candidates: int, timeout: int = 15) -> int:
    """Call LLM to pick the true #1 item."""
    import re
    from openai import OpenAI

    client = OpenAI(base_url=base_url.rstrip('/'), api_key=api_key, timeout=timeout)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=500,
        )
        text = response.choices[0].message.content or ""

        # Try JSON parsing
        match = re.search(r'"item_index"\s*:\s*(\d+)', text)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < n_candidates:
                return idx

        # Fallback: first valid number
        nums = re.findall(r'\b(\d+)\b', text)
        for n_str in nums:
            idx = int(n_str)
            if 0 <= idx < n_candidates:
                return idx
    except Exception:
        pass

    return 0  # default to algorithm's top pick


# ==================== Experiment Runner ====================

BUDGET_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
N_SEEDS = 100  # noise seeds per budget level
TOP_K = 5
RESULTS_DIR = os.path.join(script_dir, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LLM_MODELS = {
    "qwen-7b": API_KEYS["qwen-7b"][0],
    "qwen-14b": API_KEYS["qwen-14b"][0],
    "qwen-32b": API_KEYS["qwen-32b"][0],
    "qwen-72b": API_KEYS["qwen-72b"][0],
    "qwen3-max": API_KEYS["qwen3_new"][0],
}


def run_single_ranking_dfs(dataset_key: str, budget: int, seed: int) -> Dict[str, Any]:
    """Run single ranking with pure noisy algorithm."""
    dataset = DATASETS[dataset_key]
    result = noisy_rank_algorithm(dataset, budget, seed, top_k=TOP_K)

    return {
        'agent_type': 'dfs',
        'model': None,
        'prompt': None,
        'dataset': dataset_key,
        'budget': budget,
        'seed': seed,
        'algo_pick': result['algo_pick'],
        'correct': result['algo_correct'],
        'true_best': result['true_best'],
    }


def run_single_ranking_llm(dataset_key: str, budget: int, seed: int,
                            model_key: str, model_config: Dict[str, Any],
                            prompt_key: str = "standard") -> Dict[str, Any]:
    """Run single ranking with LLM selection from noisy candidates."""
    dataset = DATASETS[dataset_key]
    algo_result = noisy_rank_algorithm(dataset, budget, seed, top_k=TOP_K)

    system_prompt = SYSTEM_PROMPTS_RANKING[prompt_key]
    user_msg = format_ranking_for_llm(dataset, algo_result)

    chosen_idx = call_llm_ranking(
        model_config['base_url'], model_config['api_key'],
        model_config['model'], system_prompt, user_msg,
        len(algo_result['top_k']), timeout=15
    )

    # Check if LLM's pick is the true best
    chosen_name = algo_result['top_k'][chosen_idx][0]
    true_best = algo_result['true_best']
    llm_correct = (chosen_name == true_best)

    return {
        'agent_type': 'llm',
        'model': model_key,
        'prompt': prompt_key,
        'dataset': dataset_key,
        'budget': budget,
        'seed': seed,
        'llm_pick': chosen_name,
        'correct': llm_correct,
        'true_best': true_best,
        'algo_pick': algo_result['algo_pick'],
        'algo_correct': algo_result['algo_correct'],
    }


def run_ranking_dfs_baseline():
    """Run all DFS baseline experiments across all datasets."""
    print("  Ranking DFS baseline...")
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
        futures = []
        for dk in DATASETS:
            for budget in BUDGET_LEVELS:
                for seed in range(N_SEEDS):
                    futures.append(ex.submit(run_single_ranking_dfs, dk, budget, seed))

        done = 0
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
                done += 1
                if done % 500 == 0:
                    print(f"    DFS: {done}/{len(futures)}")
            except Exception as e:
                print(f"    DFS error: {e}")
                done += 1

    outpath = os.path.join(RESULTS_DIR, "ranking_dfs_baseline.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    Saved {len(results)} results to {outpath}")

    # Print summary
    for dk in DATASETS:
        print(f"\n    {dk}:")
        for budget in BUDGET_LEVELS:
            vals = [r['correct'] for r in results
                    if r['dataset'] == dk and r['budget'] == budget]
            if vals:
                acc = sum(vals) / len(vals)
                print(f"      budget={budget:4d}: accuracy={acc:.1%} (n={len(vals)})")

    return results


def run_ranking_llm_experiment(model_key: str, model_config: Dict[str, Any],
                                prompt_key: str = "standard"):
    """Run ranking LLM experiment for one model/prompt."""
    print(f"  Ranking LLM: model={model_key}, prompt={prompt_key}")
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = []
        for dk in DATASETS:
            for budget in BUDGET_LEVELS:
                for seed in range(N_SEEDS):
                    futures.append(ex.submit(
                        run_single_ranking_llm, dk, budget, seed,
                        model_key, model_config, prompt_key
                    ))

        done = 0
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
                done += 1
                if done % 100 == 0:
                    print(f"    {model_key}/{prompt_key}: {done}/{len(futures)}")
            except Exception as e:
                print(f"    LLM error: {e}")
                done += 1

    outpath = os.path.join(RESULTS_DIR, f"ranking_llm_{model_key}_{prompt_key}.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    Saved {len(results)} results to {outpath}")

    # Print summary
    for dk in DATASETS:
        print(f"\n    {dk}:")
        for budget in BUDGET_LEVELS:
            vals = [r['correct'] for r in results
                    if r['dataset'] == dk and r['budget'] == budget]
            if vals:
                acc = sum(vals) / len(vals)
                print(f"      budget={budget:4d}: accuracy={acc:.1%} (n={len(vals)})")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ranking Experiment: LLM vs Noisy Algorithm")
    parser.add_argument("--phase", choices=["dfs", "llm", "llm-prompts", "all"], default="all")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="standard")
    args = parser.parse_args()

    print("=" * 60)
    print("RANKING EXPERIMENT: LLM Beats Algorithm at Low Budget")
    print("=" * 60)

    if args.phase in ("dfs", "all"):
        run_ranking_dfs_baseline()

    if args.phase in ("llm", "all"):
        if args.model:
            run_ranking_llm_experiment(args.model, LLM_MODELS[args.model], args.prompt)
        else:
            for mk, mc in LLM_MODELS.items():
                run_ranking_llm_experiment(mk, mc, "standard")

    if args.phase in ("llm-prompts", "all"):
        for pk in SYSTEM_PROMPTS_RANKING:
            if pk == "standard" and args.phase == "all":
                continue  # already run above
            run_ranking_llm_experiment("qwen-32b", LLM_MODELS["qwen-32b"], pk)


if __name__ == "__main__":
    main()
