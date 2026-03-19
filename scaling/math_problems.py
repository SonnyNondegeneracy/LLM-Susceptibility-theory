"""
Deterministic Math Problem Generator

4 problem types × 50 problems each = 200 total.
Each problem has: question text, correct answer, and 4 multiple-choice options.
"""

import random
from typing import Dict, Any, List


def _generate_multiplication(seed: int) -> Dict[str, Any]:
    """Generate a multi-digit multiplication problem."""
    rng = random.Random(seed)
    a = rng.randint(12, 99)
    b = rng.randint(12, 99)
    answer = a * b

    # Generate 3 distractors (nearby wrong answers)
    distractors = set()
    attempts = 0
    while len(distractors) < 3 and attempts < 100:
        offset = rng.choice([-1, 1]) * rng.randint(1, max(1, answer // 10))
        d = answer + offset
        if d != answer and d > 0:
            distractors.add(d)
        attempts += 1
    distractors = list(distractors)[:3]

    return {
        "type": "multiplication",
        "question": f"What is {a} × {b}?",
        "answer": answer,
        "distractors": distractors,
    }


def _generate_word_problem(seed: int) -> Dict[str, Any]:
    """Generate a multi-step word problem."""
    rng = random.Random(seed)
    variant = rng.randint(0, 2)

    if variant == 0:
        # Price × quantity - discount
        price = rng.randint(5, 50)
        qty = rng.randint(3, 12)
        discount = rng.randint(1, price * qty // 4)
        answer = price * qty - discount
        question = (
            f"A store sells notebooks for ${price} each. "
            f"You buy {qty} notebooks and get a ${discount} discount. "
            f"How much do you pay in total?"
        )
    elif variant == 1:
        # (workers × hours × rate) + bonus
        workers = rng.randint(2, 8)
        hours = rng.randint(3, 10)
        rate = rng.randint(10, 30)
        bonus = rng.randint(10, 100)
        answer = workers * hours * rate + bonus
        question = (
            f"{workers} workers each work {hours} hours at ${rate}/hour. "
            f"They also receive a team bonus of ${bonus}. "
            f"What is the total payment?"
        )
    else:
        # Distance = speed × time, two legs
        speed1 = rng.randint(30, 80)
        time1 = rng.randint(2, 5)
        speed2 = rng.randint(30, 80)
        time2 = rng.randint(2, 5)
        answer = speed1 * time1 + speed2 * time2
        question = (
            f"A car travels at {speed1} km/h for {time1} hours, "
            f"then at {speed2} km/h for {time2} hours. "
            f"What is the total distance traveled in km?"
        )

    distractors = set()
    attempts = 0
    while len(distractors) < 3 and attempts < 100:
        offset = rng.choice([-1, 1]) * rng.randint(1, max(1, answer // 5))
        d = answer + offset
        if d != answer and d > 0:
            distractors.add(d)
        attempts += 1
    distractors = list(distractors)[:3]

    return {
        "type": "word_problem",
        "question": question,
        "answer": answer,
        "distractors": distractors,
    }


def _generate_expression_comparison(seed: int) -> Dict[str, Any]:
    """Generate an expression comparison problem (which is larger?)."""
    rng = random.Random(seed)

    # Two expressions, one is larger
    a1 = rng.randint(10, 50)
    b1 = rng.randint(10, 50)
    a2 = rng.randint(10, 50)
    b2 = rng.randint(10, 50)

    val1 = a1 * b1
    val2 = a2 * b2

    # Ensure they're different
    while val1 == val2:
        b2 += 1
        val2 = a2 * b2

    if val1 > val2:
        answer = val1
        question = (
            f"Which value is larger: {a1} × {b1} or {a2} × {b2}? "
            f"Give the larger value as a number."
        )
    else:
        answer = val2
        question = (
            f"Which value is larger: {a1} × {b1} or {a2} × {b2}? "
            f"Give the larger value as a number."
        )

    smaller = min(val1, val2)
    distractors = set()
    distractors.add(smaller)  # The other expression is a natural distractor
    attempts = 0
    while len(distractors) < 3 and attempts < 100:
        offset = rng.choice([-1, 1]) * rng.randint(1, max(1, answer // 10))
        d = answer + offset
        if d != answer and d > 0:
            distractors.add(d)
        attempts += 1
    distractors = list(distractors)[:3]

    return {
        "type": "expression_comparison",
        "question": question,
        "answer": answer,
        "distractors": distractors,
    }


def _generate_percentage(seed: int) -> Dict[str, Any]:
    """Generate a percentage calculation problem."""
    rng = random.Random(seed)
    variant = rng.randint(0, 1)

    if variant == 0:
        # What is X% of Y?
        pct = rng.choice([10, 15, 20, 25, 30, 40, 50, 60, 75])
        base = rng.randint(4, 40) * 20  # Ensure clean division
        answer = base * pct // 100
        question = f"What is {pct}% of {base}?"
    else:
        # X is what percent of Y?
        base = rng.randint(4, 40) * 10
        pct = rng.choice([10, 20, 25, 30, 40, 50, 60, 75, 80])
        part = base * pct // 100
        answer = pct
        question = f"{part} is what percent of {base}?"

    distractors = set()
    attempts = 0
    while len(distractors) < 3 and attempts < 100:
        offset = rng.choice([-1, 1]) * rng.randint(1, max(1, answer // 3 + 1))
        d = answer + offset
        if d != answer and d > 0:
            distractors.add(d)
        attempts += 1
    distractors = list(distractors)[:3]

    return {
        "type": "percentage",
        "question": question,
        "answer": answer,
        "distractors": distractors,
    }


GENERATORS = {
    "multiplication": _generate_multiplication,
    "word_problem": _generate_word_problem,
    "expression_comparison": _generate_expression_comparison,
    "percentage": _generate_percentage,
}

N_PER_TYPE = 50


def generate_all_problems() -> List[Dict[str, Any]]:
    """Generate all 200 math problems (50 per type) with deterministic seeds."""
    problems = []
    for ptype, gen_fn in GENERATORS.items():
        for i in range(N_PER_TYPE):
            seed = hash((ptype, i)) & 0x7FFFFFFF  # Deterministic seed per problem
            prob = gen_fn(seed)
            prob["id"] = len(problems)
            prob["seed"] = seed

            # Build multiple choice options: answer + 3 distractors, shuffled
            options = [prob["answer"]] + prob["distractors"]
            shuffle_rng = random.Random(seed + 9999)
            shuffle_rng.shuffle(options)
            prob["options"] = options
            prob["correct_option_index"] = options.index(prob["answer"])

            problems.append(prob)
    return problems


def format_math_agent(problem: Dict[str, Any]) -> str:
    """Format math problem as multiple-choice for agent mode."""
    lines = [problem["question"], ""]
    for idx, opt in enumerate(problem["options"]):
        lines.append(f"  {idx}. {opt}")
    lines.append("")
    lines.append("Which option is correct? Give the 0-based index.")
    return "\n".join(lines)


def format_math_alone(problem: Dict[str, Any]) -> str:
    """Format math problem as open-ended for LLM-alone mode."""
    return problem["question"] + "\n\nSolve this and give the numerical answer."


if __name__ == "__main__":
    problems = generate_all_problems()
    print(f"Generated {len(problems)} problems")
    for ptype in GENERATORS:
        subset = [p for p in problems if p["type"] == ptype]
        print(f"  {ptype}: {len(subset)} problems")
        # Show first example
        p = subset[0]
        print(f"    Example: {p['question']}")
        print(f"    Answer: {p['answer']}, Options: {p['options']}")
