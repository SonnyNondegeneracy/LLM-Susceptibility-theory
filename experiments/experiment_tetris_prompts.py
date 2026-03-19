"""
Tetris Prompt Variation Experiment.
Tests minimal, standard, cot, expert prompts with qwen-32b on Tetris.
"""

import json
import os
import sys
import time
import concurrent.futures
from typing import Dict, Any

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from game_engine import GameState, REWARD_FUNCTIONS, dfs_search
from llm_agent import LLMAgent, llm_play_game, SYSTEM_PROMPT, SYSTEM_PROMPT_DFS

with open(os.path.join(script_dir, 'api_key.json'), 'r') as f:
    API_KEYS = json.load(f)

TETRIS_PROMPTS = {
    "minimal": """You play Tetris. Given the board and DFS suggestions, pick a column.
Reply ONLY: {"target_col": <number>}""",

    "standard": SYSTEM_PROMPT,  # the existing standard prompt

    "cot": """You are an AI agent playing a Block Elimination game (Tetris without rotation).

THINK STEP BY STEP:
1. What piece am I placing? What is its shape?
2. What does the board look like? Where are the holes and high columns?
3. What do the DFS suggestions recommend? Why?
4. For each suggestion, what would happen? Would it create holes? Clear lines?
5. Which option minimizes holes while keeping the board flat?

After your step-by-step analysis, respond:
```json
{"thinking": "your step-by-step analysis", "target_col": <column number>}
```""",

    "expert": """You are a world-class Tetris AI. Board: 10x20, no rotation.

EXPERT STRATEGY:
- NEVER create holes (empty cells below filled cells) - this is the #1 priority
- Keep the surface as flat as possible (minimize bumpiness)
- Build towards complete rows - partial rows are wasted
- The DFS search uses mathematical optimization; trust its top suggestion unless you see a clear flaw
- Edge columns (0, 9) are dangerous - pieces there often create holes
- I-pieces are best placed to fill gaps; O-pieces need flat surfaces
- If all DFS suggestions create holes, pick the one with fewest holes

The DFS algorithm has already evaluated millions of states. Override it ONLY if you see an obvious error.

```json
{"thinking": "expert analysis", "target_col": <column number>}
```"""
}

BEAM_WIDTHS = [1, 2, 4, 8, 16, 32]
SEEDS = list(range(42, 52))
MAX_STEPS = 50
GARBAGE_LINES = 6
RESULTS_DIR = os.path.join(script_dir, "results")

MODEL_KEY = "qwen-32b"
MODEL_CONFIG = API_KEYS["qwen-32b"][0]


def run_tetris_llm_prompt(seed, beam_width, prompt_key):
    """Run single tetris game with specific prompt."""
    game = GameState(seed=seed)
    game.add_garbage_lines(GARBAGE_LINES)

    agent = LLMAgent(
        base_url=MODEL_CONFIG['base_url'],
        api_key=MODEL_CONFIG['api_key'],
        model_name=MODEL_CONFIG['model'],
        reward_function='aggressive',
        dfs_depth=3,
        dfs_width=beam_width,
        dfs_top_k=3,
        use_dfs_hint=True,
        temperature=0.1,
        max_retries=2,
        retry_delay=0.5,
        timeout=15,
        system_prompt=TETRIS_PROMPTS[prompt_key],
    )

    result = llm_play_game(game, agent, max_steps=MAX_STEPS)

    return {
        'agent_type': 'llm',
        'model': MODEL_KEY,
        'prompt': prompt_key,
        'seed': seed,
        'beam_width': beam_width,
        'lines_cleared': game.lines_cleared,
        'score': game.score,
        'steps_taken': result['steps_taken'],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    prompts_to_test = [args.prompt] if args.prompt else list(TETRIS_PROMPTS.keys())

    for prompt_key in prompts_to_test:
        if prompt_key == "standard":
            continue  # already have this data from the main experiment
        print(f"  Tetris prompt variant: {prompt_key}")
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            futures = []
            for bw in BEAM_WIDTHS:
                for seed in SEEDS:
                    futures.append(ex.submit(run_tetris_llm_prompt, seed, bw, prompt_key))

            done = 0
            for f in concurrent.futures.as_completed(futures):
                try:
                    results.append(f.result())
                    done += 1
                    if done % 10 == 0:
                        print(f"    {prompt_key}: {done}/{len(futures)}")
                except Exception as e:
                    print(f"    Error: {e}")
                    done += 1

        outpath = os.path.join(RESULTS_DIR, f"tetris_prompt_{prompt_key}.json")
        with open(outpath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"    Saved {len(results)} to {outpath}")


if __name__ == "__main__":
    main()
