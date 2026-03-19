"""
LLM AI Agent for Block Elimination Game
Connects to any OpenAI-compatible API (base_url, api_key, model_name).
Provides:
- Human/LLM-readable game state formatting
- LLM-readable DFS top-k results
- Structured prompt for LLM to choose actions
- Response parsing with retry strategy
"""

import json
import re
import time
from openai import OpenAI
from typing import Optional, Dict, Any, List, Callable
from game_engine import (
    GameState, PIECE_TYPES, PIECE_NAMES,
    REWARD_FUNCTIONS, default_reward,
    dfs_search, get_possible_placements,
    BOARD_WIDTH, BOARD_HEIGHT
)


# ==================== State Formatting for LLM ====================

def format_board_ascii(game: GameState) -> str:
    """Format the board as ASCII art for LLM to read."""
    board = game.get_board_with_piece()
    piece_chars = {0: '.', 1: 'I', 2: 'O', 3: 'T', 4: 'S', 5: 'Z', 6: 'L', 7: 'J'}
    lines = []
    lines.append("    " + "".join(f"{c:2d}" for c in range(game.width)))
    lines.append("   +" + "--" * game.width + "+")
    for r in range(game.height):
        row_str = "".join(f" {piece_chars.get(board[r][c], '?')}" for c in range(game.width))
        lines.append(f"{r:2d} |{row_str} |")
    lines.append("   +" + "--" * game.width + "+")
    return "\n ".join(lines)


def format_game_state_for_llm(game: GameState) -> str:
    """Format complete game state as a human/LLM-readable string."""
    state = game.get_state_dict()
    heights = game.get_column_heights()
    
    text = f"""=== BLOCK ELIMINATION GAME STATE ===

BOARD ({game.width}x{game.height}):
{format_board_ascii(game)}

CURRENT PIECE: {state['current_piece']['type'] if state['current_piece'] else 'None'}
NEXT PIECE: {state['next_piece_type']}

SCORE: {state['score']}
LINES CLEARED: {state['lines_cleared']}
PIECES PLACED: {state['pieces_placed']}
GAME OVER: {state['game_over']}

BOARD METRICS:
- Column Heights (col 0-{game.width-1}): {heights}
- Max Height: {max(heights) if heights else 0}
- Aggregate Height: {game.get_aggregate_height()}
- Holes: {game.count_holes()}
- Bumpiness: {game.get_bumpiness()}

VALID PLACEMENTS (target columns): {get_possible_placements(game)}
"""
    return text


def format_dfs_results_for_llm(dfs_results: List[Dict[str, Any]]) -> str:
    """Format DFS top-k results as LLM-readable text."""
    if not dfs_results:
        return "DFS SEARCH: No results available."
    
    lines = ["=== DFS SEARCH RESULTS (Top-K Action Sequences) ==="]
    for item in dfs_results:
        rank = item['rank']
        reward = item['reward']
        actions = item['actions']
        lines.append(f"DFS RESULT {rank}")
        for i, a in enumerate(actions):
            lines.append(f"  Step {i+1}: Place piece '{a['piece_type']}' at column {a['target_col']} "
                        f"(lines cleared: {a['lines_cleared']})")
        lines.append("")
    return "\n".join(lines)



# ==================== LLM Communication ====================

SYSTEM_PROMPT_DFS = """You are an AI agent playing a Block Elimination game (similar to Tetris but without rotation).

GAME RULES:
- Blocks fall from the top of a 10x20 board
- You can move blocks left/right and drop them, but CANNOT rotate them
- When a full row is filled, it is cleared and you earn points
- Game ends when blocks stack to the top

YOUR TASK:
- Analyze the current board state and metrics
- Consider the DFS search results (if provided) which suggest the top-k best placement columns based on a shallow search.
- Choose the ONE best column to place the current piece from the top-k suggestions provided in the DFS results.

RESPONSE FORMAT (you MUST follow this exactly):
```json
{
    "thinking": "Analysis of: 1) Current board state (holes, height distribution, danger zones), 2) Piece characteristics, 3) DFS suggestions evaluation, 4) Why chosen column is optimal",
    "target_col": <integer column number chosen from the DFS suggestions>
}
```

IMPORTANT:
- target_col must be one of the column numbers suggested in the "DFS SEARCH RESULTS" section.
- Consider: minimizing holes, keeping the board flat, setting up line clears, and the future consequences of each move as shown in the DFS results.
- Respond ONLY with the JSON block, no other text outside the json block.
"""


SYSTEM_PROMPT = """You are an AI agent playing a Block Elimination game (similar to Tetris but without rotation).

GAME RULES:
- Board size: 10 columns × 20 rows
- Blocks fall from the top, you can move them left/right and drop them
- CANNOT rotate pieces
- Full rows are cleared, earning points
- Game ends when blocks reach the top

YOUR TASK:
- Analyze the current board state, metrics, and piece type
- Review the DFS search results (if provided) as ONE of the information sources, but you are not obligated to follow them if you see a better move
- Use logical reasoning to select the optimal placement column
- You may choose a column from DFS suggestions OR propose a better alternative if DFS results are suboptimal

RESPONSE FORMAT (you MUST follow this exactly):
```json
{
    "thinking": "Analysis of: 1) Current board state (holes, height distribution, danger zones), 2) Piece characteristics, 3) DFS suggestions evaluation, 4) Why chosen column is optimal",
    "target_col": <integer column number (0-9, adjusted for piece width)>
}
```

DECISION CRITERIA (priority order):
1. **Survival**: Avoid creating unfillable holes or stacking too high in any column
2. **Line clears**: Prioritize moves that complete rows
3. **Board flatness**: Keep column heights as even as possible
4. **Future setup**: Create opportunities for subsequent pieces
5. **Hole minimization**: Avoid creating gaps that are hard to fill

IMPORTANT:
- DFS results show a shallow lookahead; they may miss long-term consequences
- If DFS suggestions all create holes, unevenness, or danger, use your reasoning to find a better column
- Consider the specific piece type: some pieces naturally fit certain positions better
- Respond ONLY with the JSON block, no other text outside it.
"""

def call_llm_api(
    base_url: str,
    api_key: str,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 500,
    timeout: int = 30
) -> Optional[str]:
    """Call OpenAI-compatible chat completion API using the openai SDK."""
    try:
        client = OpenAI(
            base_url=base_url.rstrip('/'),
            api_key=api_key,
            timeout=timeout,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
            raise LLMError(f"LLM API timeout after {timeout}s")
        raise LLMError(f"LLM API error: {error_msg[:300]}")


class LLMError(Exception):
    pass


def parse_llm_response(response_text: str, valid_placements: List[int]) -> Dict[str, Any]:
    """Parse LLM response and extract the action.
    
    Tries multiple strategies to extract JSON from the response.
    Validates that target_col is in valid_placements.
    """
    # Strategy 1: Find JSON block in ```json ... ``` markers
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if 'target_col' in parsed:
                col = int(parsed['target_col'])
                if col in valid_placements:
                    return {
                        'target_col': col,
                        'thinking': parsed.get('thinking', ''),
                        'raw_response': response_text,
                        'parse_method': 'json_block'
                    }
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Strategy 2: Find any JSON object in the response
    json_match = re.search(r'\{[^{}]*"target_col"\s*:\s*(\d+)[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            col = int(parsed['target_col'])
            if col in valid_placements:
                return {
                    'target_col': col,
                    'thinking': parsed.get('thinking', ''),
                    'raw_response': response_text,
                    'parse_method': 'json_object'
                }
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Strategy 3: Find just a number that matches "target_col": N or "column N"
    col_match = re.search(r'(?:target_col|column)\s*[:\s]\s*(\d+)', response_text, re.IGNORECASE)
    if col_match:
        col = int(col_match.group(1))
        if col in valid_placements:
            return {
                'target_col': col,
                'thinking': '',
                'raw_response': response_text,
                'parse_method': 'regex_extract'
            }
    
    # Strategy 4: Find any number in the response that's a valid placement
    numbers = re.findall(r' (\d+) ', response_text)
    for num_str in numbers:
        num = int(num_str)
        if num in valid_placements:
            return {
                'target_col': num,
                'thinking': '',
                'raw_response': response_text,
                'parse_method': 'fallback_number'
            }
    
    raise LLMError(f"Could not parse valid target_col from LLM response. Valid: {valid_placements}. Response: {response_text[:300]}")


# ==================== LLM Agent Class ====================

class LLMAgent:
    """LLM-based AI agent that uses an OpenAI-compatible API to play the game."""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        reward_function: str = 'default',
        dfs_depth: int = 2,
        dfs_width: int = 10,
        dfs_top_k: int = 5,
        use_dfs_hint: bool = True,
        temperature: float = 0.3,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        system_prompt: Optional[str] = None,
        timeout: int = 30,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.reward_function_name = reward_function
        self.reward_fn = REWARD_FUNCTIONS.get(reward_function, default_reward)
        self.dfs_depth = dfs_depth
        self.dfs_width = dfs_width
        self.dfs_top_k = dfs_top_k
        self.use_dfs_hint = use_dfs_hint
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.timeout = timeout
        self.name = f"LLMAgent({model_name})"
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        self.step_log: List[Dict[str, Any]] = []
        
        # Error tracking
        self.error_counts = {
            'timeout': 0,
            'api_error': 0,
            'parse_error': 0,
            'total_requests': 0
        }
    
    def _build_user_message(self, game: GameState) -> str:
        """Build the user message with game state and optional DFS hints."""
        parts = [format_game_state_for_llm(game)]
        
        if self.use_dfs_hint:
            try:
                # Use dfs_search (which is Beam Search)
                beam_result = dfs_search(game, self.reward_fn, self.dfs_depth, self.dfs_top_k, self.dfs_width)
                dfs_text = format_dfs_results_for_llm(beam_result.to_dict())
                parts.append(dfs_text)
            except Exception as e:
                parts.append(f"DFS SEARCH: Error computing - {str(e)}")
        
        parts.append("Please analyze the board and choose the best target column for the current piece. "
                     "Respond with the JSON format specified in the system prompt.")
        
        return "\n".join(parts)
    
    def choose_action(self, game: GameState) -> Optional[Dict[str, Any]]:
        """
        Ask the LLM to choose an action. Returns dict with target_col and metadata.
        Implements retry strategy on parse failures or API errors.
        """
        if game.game_over or game.current_piece is None:
            return None
        
        valid_placements = get_possible_placements(game)
        if not valid_placements:
            return None
        
        user_message = self._build_user_message(game)
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt}
                ]
                
                # Add conversation context (last few exchanges for continuity)
                for entry in self.conversation_history[-4:]:
                    messages.append(entry)
                
                # Current request
                current_user_msg = user_message
                if attempt > 0 and last_error:
                    current_user_msg += (
                        f"⚠️ RETRY (attempt {attempt + 1}/{self.max_retries}): "
                        f"Previous response could not be parsed. Error: {last_error}"
                        f"Please respond with ONLY a valid JSON object containing 'target_col' "
                        f"as one of these valid columns: {valid_placements}"
                    )
                
                messages.append({"role": "user", "content": current_user_msg})
                
                # Call LLM
                start_time = time.time()
                self.error_counts['total_requests'] += 1
                response_text = call_llm_api(
                    self.base_url, self.api_key, self.model_name,
                    messages, self.temperature, timeout=self.timeout
                )
                api_time = time.time() - start_time
                
                # Parse response
                result = parse_llm_response(response_text, valid_placements)
                result['attempt'] = attempt + 1
                result['api_time'] = api_time
                result['model'] = self.model_name
                
                # Save to conversation history
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                # Keep conversation history manageable
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-10:]
                
                # Log this step
                self.step_log.append({
                    'step': len(self.step_log) + 1,
                    'piece_type': game.current_piece.piece_type,
                    'target_col': result['target_col'],
                    'thinking': result.get('thinking', ''),
                    'parse_method': result.get('parse_method', ''),
                    'attempt': result['attempt'],
                    'api_time': result['api_time'],
                    'score_before': game.score,
                })
                
                return result
                
            except LLMError as e:
                last_error = str(e)
                if 'timeout' in last_error.lower():
                    self.error_counts['timeout'] += 1
                elif 'parse' in last_error.lower() or 'target_col' in last_error.lower():
                    self.error_counts['parse_error'] += 1
                else:
                    self.error_counts['api_error'] += 1
                    
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                continue
        
        # All retries failed - fallback to Beam Search best action
        try:
            # Use dfs_search (which is Beam Search)
            beam_result = dfs_search(game, self.reward_fn, 1, 1, 1)
            if beam_result.top_k and beam_result.top_k[0][1]:
                fallback_col = beam_result.top_k[0][1][0]['target_col']
                return {
                    'target_col': fallback_col,
                    'thinking': 'LLM failed, using Beam Search fallback',
                    'parse_method': 'beam_search_fallback',
                    'attempt': self.max_retries,
                    'api_time': 0,
                    'model': self.model_name,
                }
        except:
            pass
        
        # Ultimate fallback: first valid placement
        if valid_placements:
            return {
                'target_col': valid_placements[len(valid_placements) // 2],
                'thinking': 'All strategies failed, using middle column',
                'parse_method': 'ultimate_fallback',
                'attempt': self.max_retries,
                'api_time': 0,
                'model': self.model_name,
            }
        
        return None
    
    def get_step_log(self) -> List[Dict[str, Any]]:
        """Get the log of all steps taken by this agent."""
        return self.step_log
        
    def get_error_stats(self) -> Dict[str, int]:
        """Get statistics about errors encountered."""
        return self.error_counts.copy()
    
    def reset_history(self):
        """Reset conversation history and step log."""
        self.conversation_history = []
        self.step_log = []


# ==================== Convenience Functions ====================

def create_llm_agent(config: Dict[str, Any]) -> LLMAgent:
    """Create an LLM agent from a config dict."""
    return LLMAgent(
        base_url=config['base_url'],
        api_key=config['api_key'],
        model_name=config['model_name'],
        reward_function=config.get('reward_function', 'default'),
        dfs_depth=config.get('dfs_depth', 2),
        dfs_top_k=config.get('dfs_top_k', 5),
        use_dfs_hint=config.get('use_dfs_hint', True),
        temperature=config.get('temperature', 0.3),
        max_retries=config.get('max_retries', 3),
        retry_delay=config.get('retry_delay', 1.0),
        system_prompt=config.get('system_prompt', None),
        timeout=config.get('timeout', 30),
    )


def llm_play_game(
    game: GameState,
    agent: LLMAgent,
    max_steps: int = 200,
    step_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """Have an LLM agent play a full game.
    
    Args:
        game: GameState to play
        agent: LLMAgent instance
        max_steps: Maximum number of steps
        step_callback: Optional callback(step, game, action_result) called after each step
    
    Returns:
        Summary dict with results and agent log
    """
    start_time = time.time()
    steps = 0
    score_history = []
    
    while not game.game_over and steps < max_steps:
        action = agent.choose_action(game)
        if action is None:
            break
        
        target_col = action['target_col']
        result = game.drop_and_place(target_col)
        steps += 1
        
        step_info = {
            'step': steps,
            'score': game.score,
            'lines_cleared': game.lines_cleared,
            'holes': game.count_holes(),
            'aggregate_height': game.get_aggregate_height(),
            'bumpiness': game.get_bumpiness(),
            'target_col': target_col,
            'thinking': action.get('thinking', ''),
            'parse_method': action.get('parse_method', ''),
            'api_time': action.get('api_time', 0),
        }
        score_history.append(step_info)
        
        if step_callback:
            step_callback(steps, game, step_info)
    
    elapsed = time.time() - start_time
    
    return {
        'model': agent.model_name,
        'reward_function': agent.reward_function_name,
        'use_dfs_hint': agent.use_dfs_hint,
        'steps_taken': steps,
        'elapsed_time': elapsed,
        'summary': game.export_summary(),
        'score_history': score_history,
        'agent_log': agent.get_step_log(),
        'error_stats': agent.get_error_stats(),
    }
