"""
Hard Math Problem Generator for Phase 3

Two sources:
  1. AIME (American Invitational Mathematics Examination) 2024-2026 from HuggingFace
     - 90 problems, integer answers 0-999
     - Authoritative, well-known benchmark, low contamination for 2025-2026
     - Target: ~15-35% single-attempt accuracy for non-reasoning 32B models
  2. Fallback: self-generated problems (120 total) if HuggingFace unavailable

All answers are integers.
"""

import os
import re
import json
import random
import math
from typing import Dict, Any, List

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aime_problems_cache.json")


def _clean_latex(text: str) -> str:
    """Lightly clean LaTeX for LLM consumption — keep math notation readable."""
    # Remove common LaTeX wrappers but keep the math content
    text = text.replace("\\\\", "\n")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\(", "").replace("\\)", "")
    # Keep $...$ notation as-is since LLMs understand it
    return text.strip()


def _load_aime_from_huggingface() -> List[Dict[str, Any]]:
    """Load AIME 2024+2025+2026 from HuggingFace (90 problems)."""
    from datasets import load_dataset

    problems = []
    pid = 0

    # AIME 2024
    ds = load_dataset("Maxwell-Jia/AIME_2024")
    for row in ds["train"]:
        problems.append({
            "id": pid,
            "type": "aime_2024",
            "source": row.get("ID", f"2024-{pid}"),
            "question": _clean_latex(row["Problem"]),
            "answer": int(row["Answer"]),
        })
        pid += 1

    # AIME 2025
    ds = load_dataset("MathArena/aime_2025")
    for row in ds["train"]:
        problems.append({
            "id": pid,
            "type": "aime_2025",
            "source": f"2025-{row['problem_idx']}",
            "question": _clean_latex(row["problem"]),
            "answer": int(row["answer"]),
        })
        pid += 1

    # AIME 2026
    ds = load_dataset("MathArena/aime_2026")
    for row in ds["train"]:
        problems.append({
            "id": pid,
            "type": "aime_2026",
            "source": f"2026-{row['problem_idx']}",
            "question": _clean_latex(row["problem"]),
            "answer": int(row["answer"]),
        })
        pid += 1

    return problems


def _save_cache(problems: List[Dict[str, Any]]):
    with open(CACHE_FILE, 'w') as f:
        json.dump(problems, f, indent=1)


def _load_cache() -> List[Dict[str, Any]]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return []


def generate_all_problems(use_aime: bool = True) -> List[Dict[str, Any]]:
    """Load AIME benchmark (90 problems) or fall back to self-generated (120).

    Args:
        use_aime: If True, try AIME from cache/HuggingFace first.
    """
    if use_aime:
        # Try cache first
        cached = _load_cache()
        if cached:
            return cached

        # Download from HuggingFace
        try:
            problems = _load_aime_from_huggingface()
            _save_cache(problems)
            return problems
        except Exception as e:
            print(f"  WARNING: Could not load AIME from HuggingFace: {e}")
            print(f"  Falling back to self-generated problems...")

    return _generate_fallback_problems()


# ==================== Fallback: Self-generated problems ====================

def _generate_fallback_problems() -> List[Dict[str, Any]]:
    """Generate 120 hard math problems as fallback."""
    generators = {
        "big_multiplication": _gen_big_mult,
        "modular_arithmetic": _gen_mod_arith,
        "multi_step_word": _gen_word,
        "combinatorics": _gen_comb,
        "counting": _gen_counting,
        "linear_system": _gen_linear,
    }
    problems = []
    for ptype, gen_fn in generators.items():
        for i in range(20):
            seed = hash((ptype, i)) & 0x7FFFFFFF
            prob = gen_fn(seed)
            prob["id"] = len(problems)
            prob["seed"] = seed
            problems.append(prob)
    return problems


def _gen_big_mult(seed):
    rng = random.Random(seed)
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    return {"type": "big_multiplication", "question": f"What is {a} × {b}?", "answer": a * b}

def _gen_mod_arith(seed):
    rng = random.Random(seed)
    primes = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    a, b, p = rng.randint(100, 999), rng.randint(100, 999), rng.choice(primes)
    return {"type": "modular_arithmetic", "question": f"What is ({a} × {b}) mod {p}?", "answer": (a * b) % p}

def _gen_word(seed):
    rng = random.Random(seed)
    w, h, r = rng.randint(3, 12), rng.randint(4, 10), rng.randint(15, 45)
    tax_pct, bonus = rng.randint(5, 20), rng.randint(50, 300)
    gross = w * h * r
    answer = gross - gross * tax_pct // 100 + bonus
    return {"type": "multi_step_word",
            "question": f"A factory has {w} workers each working {h} hours at ${r}/hour. "
                        f"A {tax_pct}% tax is deducted (round down), then a ${bonus} bonus is added. "
                        f"What is the final amount?",
            "answer": answer}

def _gen_comb(seed):
    rng = random.Random(seed)
    n = rng.randint(8, 15)
    k = rng.randint(3, min(6, n - 2))
    return {"type": "combinatorics", "question": f"What is C({n}, {k})?", "answer": math.comb(n, k)}

def _gen_counting(seed):
    rng = random.Random(seed)
    target = rng.randint(10, 20)
    count = sum(1 for n in range(100, 1000) if sum(int(d) for d in str(n)) == target)
    return {"type": "counting",
            "question": f"How many 3-digit numbers (100-999) have digits that sum to {target}?",
            "answer": count}

def _gen_linear(seed):
    rng = random.Random(seed)
    x_sol, y_sol = rng.randint(-10, 10), rng.randint(-10, 10)
    for _ in range(100):
        a = rng.randint(2, 9) * rng.choice([-1, 1])
        b = rng.randint(2, 9) * rng.choice([-1, 1])
        d = rng.randint(2, 9) * rng.choice([-1, 1])
        e = rng.randint(2, 9) * rng.choice([-1, 1])
        if a * e - b * d != 0:
            break
    c, f = a * x_sol + b * y_sol, d * x_sol + e * y_sol
    return {"type": "linear_system",
            "question": f"Solve for x: {a}x + {b}y = {c}, {d}x + {e}y = {f}",
            "answer": x_sol}


if __name__ == "__main__":
    problems = generate_all_problems()
    print(f"Generated {len(problems)} problems")

    # Group by type
    by_type = {}
    for p in problems:
        by_type.setdefault(p["type"], []).append(p)

    for ptype, plist in sorted(by_type.items()):
        print(f"\n  {ptype}: {len(plist)} problems")
        for p in plist[:2]:
            q_short = p["question"][:80] + ("..." if len(p["question"]) > 80 else "")
            print(f"    Q: {q_short}")
            print(f"    A: {p['answer']}")
