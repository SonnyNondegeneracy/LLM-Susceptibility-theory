"""
Phase 3 Experiment: LLM+LLM Agent vs Majority Vote Scaling

Two-phase design:
  Step 1 (Generation): For each (model, problem), call LLM k_max=21 times at temp=0.7
  Step 2 (Evaluation): For each k in [1,3,5,9,15,21], compare:
    - Majority Vote: pick most common answer among k responses
    - LLM+LLM Agent: deduplicate answers → LLM selects from unique options

5 models × 120 problems × 21 = 12,600 generation calls
Agent selection: ≤ 5 × 120 × 6 = 3,600 calls (upper bound)
"""

import json
import os
import sys
import re
import time
import random
import argparse
import concurrent.futures
from typing import Dict, Any, List, Optional
from collections import Counter

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

from hard_math_problems import generate_all_problems

# Load API keys
with open(os.path.join(parent_dir, 'api_key.json'), 'r') as f:
    API_KEYS = json.load(f)

RESULTS_DIR = os.path.join(script_dir, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model configs
MODEL_ORDER = ["qwen-7b", "qwen-14b", "qwen-32b", "qwen-72b", "qwen3-max"]
MODEL_PARAMS = {
    "qwen-7b": 7e9, "qwen-14b": 14e9, "qwen-32b": 32e9,
    "qwen-72b": 72e9, "qwen3-max": 200e9,
}
LLM_MODELS = {
    "qwen-7b": API_KEYS["qwen-7b"][0],
    "qwen-14b": API_KEYS["qwen-14b"][0],
    "qwen-32b": API_KEYS["qwen-32b"][0],
    "qwen-72b": API_KEYS["qwen-72b"][0],
    "qwen3-max": API_KEYS["qwen3_new"][0],
}

# Experiment config
K_VALUES = [1, 3, 5, 9, 15, 21]
K_MAX = 21
TEMPERATURE = 0.7
MAX_WORKERS = 12

GENERATION_CHECKPOINT = os.path.join(RESULTS_DIR, "phase3_generations.json")
RESULTS_FILE = os.path.join(RESULTS_DIR, "phase3_results.json")


# ==================== LLM Call Helpers ====================

def call_llm_generic(base_url: str, api_key: str, model_name: str,
                     system_prompt: str, user_message: str,
                     temperature: float = 0.7, timeout: int = 60) -> str:
    """Generic LLM call with configurable temperature."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url.rstrip('/'), api_key=api_key, timeout=timeout)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=1500,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {e}"


def parse_number(text: str) -> Optional[float]:
    """Extract a numerical answer from LLM response.

    Strategy: try high-confidence patterns first (boxed, explicit 'answer is'),
    then fall back to the LAST number in the response (which is usually the final answer).
    """
    if text.startswith("ERROR:"):
        return None

    # High-priority: \boxed{N}
    m = re.search(r'\\boxed\{(-?\d+(?:\.\d+)?)\}', text)
    if m:
        return float(m.group(1))

    # Explicit final answer markers (search ALL, take the LAST one)
    final_patterns = [
        r'(?:final\s+answer|answer|result|therefore|thus|so)\s*(?:is|=|:)\s*\**\s*(-?\d+(?:\.\d+)?)',
        r'\*\*(-?\d+(?:\.\d+)?)\*\*\s*$',
    ]
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if matches:
            try:
                return float(matches[-1].group(1))
            except ValueError:
                continue

    # Fallback: the LAST number in the text (most likely the final answer)
    all_nums = list(re.finditer(r'(?<![a-zA-Z])(-?\d+(?:\.\d+)?)(?![a-zA-Z\d])', text))
    if all_nums:
        try:
            return float(all_nums[-1].group(1))
        except ValueError:
            pass
    return None


def parse_selection(text: str, n_options: int) -> Optional[int]:
    """Parse LLM's selection from numbered options."""
    if text.startswith("ERROR:"):
        return None
    # Look for patterns like "Option 2", "I choose 2", "2", "#2"
    patterns = [
        r'(?:option|choice|select|choose|pick|answer)\s*(?:#|number)?\s*(\d+)',
        r'#(\d+)',
        r'(?:^|\n)\s*(\d+)\s*(?:$|\n|\.)',
        r'(\d+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= n_options:
                return idx
    return None


# ==================== Step 1: Generation ====================

GENERATION_SYSTEM = """You are a math problem solver. Solve the given problem step by step and provide the numerical answer. Give your final answer as a single number."""

GENERATION_USER = """Solve this math problem. Show your work, then state your final numerical answer clearly.

{question}

Your final answer (a single number):"""


def generate_single(model_key: str, problem: Dict, attempt: int) -> Dict:
    """Generate a single LLM response for a problem."""
    cfg = LLM_MODELS[model_key]
    prompt = GENERATION_USER.format(question=problem["question"])
    raw = call_llm_generic(
        cfg["base_url"], cfg["api_key"], cfg["model"],
        GENERATION_SYSTEM, prompt, temperature=TEMPERATURE
    )
    parsed = parse_number(raw)
    return {
        "model": model_key,
        "problem_id": problem["id"],
        "attempt": attempt,
        "raw_response": raw[:500],  # truncate for storage
        "parsed_answer": parsed,
    }


def run_generation(problems: List[Dict], models: List[str] = None):
    """Step 1: Generate k_max responses for each (model, problem)."""
    if models is None:
        models = MODEL_ORDER

    # Load existing checkpoint
    generations = {}
    if os.path.exists(GENERATION_CHECKPOINT):
        with open(GENERATION_CHECKPOINT, 'r') as f:
            existing = json.load(f)
        for g in existing:
            key = (g["model"], g["problem_id"], g["attempt"])
            generations[key] = g
        print(f"  Loaded {len(generations)} existing generations")

    # Determine missing work
    tasks = []
    for model_key in models:
        for prob in problems:
            for attempt in range(K_MAX):
                key = (model_key, prob["id"], attempt)
                if key not in generations:
                    tasks.append((model_key, prob, attempt))

    if not tasks:
        print("  All generations already complete!")
        return list(generations.values())

    print(f"  {len(tasks)} generation calls needed ({len(generations)} cached)")

    done = 0
    total = len(tasks)
    save_interval = 500  # checkpoint every 500 completions
    last_save = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for model_key, prob, attempt in tasks:
            f = ex.submit(generate_single, model_key, prob, attempt)
            futures[f] = (model_key, prob["id"], attempt)

        for f in concurrent.futures.as_completed(futures):
            try:
                result = f.result()
                key = (result["model"], result["problem_id"], result["attempt"])
                generations[key] = result
                done += 1
                if done % 200 == 0:
                    elapsed_pct = done / total * 100
                    print(f"    Generation: {done}/{total} ({elapsed_pct:.1f}%)")
                # Periodic checkpoint
                if done - last_save >= save_interval:
                    _save_generations(list(generations.values()))
                    last_save = done
            except Exception as e:
                model_key, pid, attempt = futures[f]
                print(f"    Error {model_key} p{pid} a{attempt}: {e}")
                done += 1

    all_gens = list(generations.values())
    _save_generations(all_gens)
    return all_gens


def _save_generations(generations: List[Dict]):
    """Save generation checkpoint."""
    with open(GENERATION_CHECKPOINT, 'w') as f:
        json.dump(generations, f, indent=1)
    print(f"    Checkpoint saved: {len(generations)} generations")


# ==================== Step 2: Evaluation ====================

def answers_match(a: float, b: float, tol: float = 0.5) -> bool:
    """Check if two answers match within tolerance."""
    return abs(a - b) < tol


def majority_vote(answers: List[float]) -> Optional[float]:
    """Pick most common answer. Random tie-break."""
    if not answers:
        return None
    # Group by tolerance
    groups = []
    for ans in answers:
        found = False
        for g in groups:
            if answers_match(ans, g[0]):
                g.append(ans)
                found = True
                break
        if not found:
            groups.append([ans])
    # Find largest group
    max_size = max(len(g) for g in groups)
    largest = [g for g in groups if len(g) == max_size]
    chosen_group = random.choice(largest)
    return chosen_group[0]


def deduplicate_answers(answers: List[float]) -> List[float]:
    """Deduplicate answers within tolerance, return unique representatives."""
    unique = []
    for ans in answers:
        if not any(answers_match(ans, u) for u in unique):
            unique.append(ans)
    return unique


AGENT_SELECT_SYSTEM = """You are evaluating candidate answers to a math problem. Choose the correct answer from the options provided. Think step by step about which answer is most likely correct, then state your selection."""

AGENT_SELECT_USER = """Math problem: {question}

Candidate answers (from multiple solution attempts):
{options_text}

Which option number is the correct answer? Analyze the problem and select the best answer. State your choice as "Option N"."""


def agent_select(model_key: str, problem: Dict, unique_answers: List[float]) -> Optional[float]:
    """LLM selects from deduplicated options."""
    if len(unique_answers) == 1:
        return unique_answers[0]

    options_text = "\n".join(f"  Option {i+1}: {int(a) if a == int(a) else a}"
                            for i, a in enumerate(unique_answers))

    cfg = LLM_MODELS[model_key]
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

    # Fallback: try to parse a number from response
    parsed = parse_number(raw)
    if parsed is not None:
        # Find closest match
        for u in unique_answers:
            if answers_match(parsed, u):
                return u
    return None


def evaluate_at_k(generations: List[Dict], problems: List[Dict], k: int,
                  models: List[str] = None) -> List[Dict]:
    """Evaluate both methods at a given k."""
    if models is None:
        models = MODEL_ORDER

    # Index generations by (model, problem_id) → list of parsed answers
    gen_index = {}
    for g in generations:
        key = (g["model"], g["problem_id"])
        if key not in gen_index:
            gen_index[key] = []
        gen_index[key].append(g)

    # Sort by attempt number and take first k
    for key in gen_index:
        gen_index[key].sort(key=lambda x: x["attempt"])

    prob_by_id = {p["id"]: p for p in problems}
    results = []

    # Majority vote (no extra LLM calls)
    for model_key in models:
        for prob in problems:
            key = (model_key, prob["id"])
            gens = gen_index.get(key, [])[:k]
            answers = [g["parsed_answer"] for g in gens if g["parsed_answer"] is not None]

            mv_answer = majority_vote(answers) if answers else None
            correct = mv_answer is not None and answers_match(mv_answer, prob["answer"])

            results.append({
                "method": "majority_vote",
                "model": model_key,
                "problem_id": prob["id"],
                "problem_type": prob["type"],
                "k": k,
                "answer": mv_answer,
                "true_answer": prob["answer"],
                "correct": correct,
                "n_valid": len(answers),
                "n_unique": len(deduplicate_answers(answers)) if answers else 0,
            })

    # Agent selection (may need LLM calls for multi-option cases)
    agent_tasks = []
    for model_key in models:
        for prob in problems:
            key = (model_key, prob["id"])
            gens = gen_index.get(key, [])[:k]
            answers = [g["parsed_answer"] for g in gens if g["parsed_answer"] is not None]
            unique = deduplicate_answers(answers) if answers else []
            agent_tasks.append((model_key, prob, answers, unique))

    # Run agent selections in parallel
    agent_results_map = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for model_key, prob, answers, unique in agent_tasks:
            task_key = (model_key, prob["id"])
            if not unique:
                agent_results_map[task_key] = None
            elif len(unique) == 1:
                agent_results_map[task_key] = unique[0]
            else:
                f = ex.submit(agent_select, model_key, prob, unique)
                futures[f] = task_key

        done = 0
        for f in concurrent.futures.as_completed(futures):
            try:
                agent_results_map[futures[f]] = f.result()
                done += 1
                if done % 100 == 0:
                    print(f"      Agent selection k={k}: {done}/{len(futures)}")
            except Exception as e:
                agent_results_map[futures[f]] = None
                done += 1

    for model_key, prob, answers, unique in agent_tasks:
        task_key = (model_key, prob["id"])
        ag_answer = agent_results_map.get(task_key)
        correct = ag_answer is not None and answers_match(ag_answer, prob["answer"])

        results.append({
            "method": "agent",
            "model": model_key,
            "problem_id": prob["id"],
            "problem_type": prob["type"],
            "k": k,
            "answer": ag_answer,
            "true_answer": prob["answer"],
            "correct": correct,
            "n_valid": len(answers),
            "n_unique": len(unique),
        })

    return results


# ==================== Calibration ====================

def run_calibration(problems: List[Dict]):
    """Quick calibration: 5 problems/type on qwen-72b, check single-attempt accuracy."""
    print("\n  === CALIBRATION MODE ===")
    model_key = "qwen-72b"
    cfg = LLM_MODELS[model_key]

    # Sample 5 per type
    by_type = {}
    for p in problems:
        by_type.setdefault(p["type"], []).append(p)

    total_correct = 0
    total_count = 0

    for ptype, plist in by_type.items():
        sample = plist[:5]
        correct = 0
        for prob in sample:
            prompt = GENERATION_USER.format(question=prob["question"])
            raw = call_llm_generic(
                cfg["base_url"], cfg["api_key"], cfg["model"],
                GENERATION_SYSTEM, prompt, temperature=0.1
            )
            parsed = parse_number(raw)
            is_correct = parsed is not None and answers_match(parsed, prob["answer"])
            correct += is_correct
            status = "✓" if is_correct else "✗"
            print(f"    {status} {ptype}: {prob['question'][:60]}... "
                  f"got={parsed}, expected={prob['answer']}")

        acc = correct / len(sample) * 100
        print(f"    {ptype}: {correct}/{len(sample)} = {acc:.0f}%")
        total_correct += correct
        total_count += len(sample)

    overall = total_correct / total_count * 100
    print(f"\n  Overall: {total_correct}/{total_count} = {overall:.1f}%")
    print(f"  Target: 30-60%. {'✓ In range!' if 30 <= overall <= 60 else '✗ Outside range'}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Agent vs Majority Vote")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration only")
    parser.add_argument("--eval-only", action="store_true", help="Skip generation, evaluate only")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of models")
    args = parser.parse_args()

    problems = generate_all_problems()
    print(f"Loaded {len(problems)} hard math problems")

    models = args.models if args.models else MODEL_ORDER

    if args.calibrate:
        run_calibration(problems)
        return

    # Step 1: Generation
    if args.eval_only:
        print("\n  Loading existing generations...")
        if not os.path.exists(GENERATION_CHECKPOINT):
            print("  ERROR: No generation checkpoint found. Run without --eval-only first.")
            return
        with open(GENERATION_CHECKPOINT, 'r') as f:
            generations = json.load(f)
        print(f"  Loaded {len(generations)} generations")
    else:
        print("\n" + "=" * 60)
        print("STEP 1: GENERATION")
        print("=" * 60)
        t0 = time.time()
        generations = run_generation(problems, models)
        elapsed = time.time() - t0
        print(f"  Generation complete: {len(generations)} total in {elapsed:.0f}s")

    # Step 2: Evaluation
    print("\n" + "=" * 60)
    print("STEP 2: EVALUATION")
    print("=" * 60)

    all_results = []
    for k in K_VALUES:
        print(f"\n  Evaluating k={k}...")
        t0 = time.time()
        results_k = evaluate_at_k(generations, problems, k, models)
        elapsed = time.time() - t0

        # Quick summary
        for method in ["majority_vote", "agent"]:
            for model_key in models:
                subset = [r for r in results_k
                          if r["method"] == method and r["model"] == model_key]
                acc = sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0
                print(f"    k={k:2d} {method:>14s} {model_key:>10s}: {acc:.1f}%")

        all_results.extend(results_k)
        print(f"    k={k} done in {elapsed:.0f}s")

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=1)
    print(f"\n  Results saved to {RESULTS_FILE}")

    # Sanity check: k=1 should be identical
    print("\n  Sanity check (k=1 agent == majority_vote):")
    for model_key in models:
        mv = [r for r in all_results if r["k"] == 1 and r["method"] == "majority_vote" and r["model"] == model_key]
        ag = [r for r in all_results if r["k"] == 1 and r["method"] == "agent" and r["model"] == model_key]
        mv_acc = sum(r["correct"] for r in mv) / len(mv) * 100 if mv else 0
        ag_acc = sum(r["correct"] for r in ag) / len(ag) * 100 if ag else 0
        match = "✓" if abs(mv_acc - ag_acc) < 0.1 else "✗"
        print(f"    {match} {model_key}: MV={mv_acc:.1f}%, Agent={ag_acc:.1f}%")

    print("\n" + "=" * 60)
    print("PHASE 3 EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
