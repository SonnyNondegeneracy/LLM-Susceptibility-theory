"""
Core Game Engine for Block Elimination Game (Tetris-like)

This module provides the core logic for the block elimination game, including:
- Game state management (board, pieces, score)
- Piece movement and collision detection
- Line clearing and scoring
- Game over condition check
- Interface for AI agents to interact with the game
- Implementation of common AI agents (Greedy, Random, DFS/Beam Search)
- DFS/Beam Search algorithm for finding optimal moves

Usage:
    The `GameState` class is the main entry point for interacting with the game.
    You can create a `GameState` instance and use its methods to control the game,
    such as `move_left()`, `move_right()`, `move_down()`, and `drop_and_place()`.

    AI agents can be used to automatically play the game. They implement a
    `choose_action(state)` method that takes the current `GameState` and returns
    the best column to place the current piece.

    The `dfs_search` function can be used to perform a depth-first search (or beam search)
    to find the best sequence of moves based on a given reward function.
"""

import copy
import random
import time
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable

# Board dimensions
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# Piece definitions (each piece is a list of (row, col) offsets from anchor)
# Standard Tetris-like pieces, no rotation
PIECE_TYPES = {
    "I": {
        "shape": [(0, 0), (0, 1), (0, 2), (0, 3)],
        "color": "#00FFFF",
        "id": 1
    },
    "I_v": {
        "shape": [(0, 0), (1, 0), (2, 0), (3, 0)],
        "color": "#00FFFF",
        "id": 1
    },
    "O": {
        "shape": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "color": "#FFFF00",
        "id": 2
    },
    "T": {
        "shape": [(0, 0), (0, 1), (0, 2), (1, 1)],
        "color": "#800080",
        "id": 3
    },
    "T_d": {
        "shape": [(0, 1), (1, 0), (1, 1), (1, 2)],
        "color": "#800080",
        "id": 3
    },
    "T_l": {
        "shape": [(0, 1), (1, 0), (1, 1), (2, 1)],
        "color": "#800080",
        "id": 3
    },
    "T_r": {
        "shape": [(0, 0), (1, 0), (1, 1), (2, 0)],
        "color": "#800080",
        "id": 3
    },
    "S": {
        "shape": [(0, 1), (0, 2), (1, 0), (1, 1)],
        "color": "#00FF00",
        "id": 4
    },
    "S_v": {
        "shape": [(0, 0), (1, 0), (1, 1), (2, 1)],
        "color": "#00FF00",
        "id": 4
    },
    "Z": {
        "shape": [(0, 0), (0, 1), (1, 1), (1, 2)],
        "color": "#FF0000",
        "id": 5
    },
    "Z_v": {
        "shape": [(0, 1), (1, 1), (1, 0), (2, 0)],
        "color": "#FF0000",
        "id": 5
    },
    "L": {
        "shape": [(0, 0), (1, 0), (2, 0), (2, 1)],
        "color": "#FFA500",
        "id": 6
    },
    "L_l": {
        "shape": [(0, 0), (0, 1), (0, 2), (1, 0)],
        "color": "#FFA500",
        "id": 6
    },
    "L_r": {
        "shape": [(0, 2), (1, 0), (1, 1), (1, 2)],
        "color": "#FFA500",
        "id": 6
    },
    "L_u": {
        "shape": [(0, 0), (0, 1), (1, 1), (2, 1)],
        "color": "#FFA500",
        "id": 6
    },
    "J": {
        "shape": [(0, 1), (1, 1), (2, 0), (2, 1)],
        "color": "#0000FF",
        "id": 7
    },
    "J_l": {
        "shape": [(0, 0), (1, 0), (1, 1), (1, 2)],
        "color": "#0000FF",
        "id": 7
    },
    "J_r": {
        "shape": [(0, 0), (0, 1), (0, 2), (1, 2)],
        "color": "#0000FF",
        "id": 7
    },
    "J_u": {
        "shape": [(0, 0), (1, 0), (2, 0), (0, 1)],
        "color": "#0000FF",
        "id": 7
    },
}

PIECE_NAMES = list(PIECE_TYPES.keys())


class Piece:
    """
    Represents a falling piece in the game.

    Attributes:
        piece_type (str): The type of the piece ('I', 'O', 'T', 'S', 'Z', 'L', 'J').
        shape (list): A list of (row, col) offsets defining the piece's shape.
        color (str): The color of the piece.
        piece_id (int): A unique ID for the piece type.
        width (int): The width of the piece in cells.
        row (int): The current row of the piece's anchor point on the board.
        col (int): The current column of the piece's anchor point on the board.
    """

    def __init__(self, piece_type: str, col: Optional[int] = None):
        """
        Initializes a new Piece.

        Args:
            piece_type (str): The type of the piece.
            col (Optional[int]): The starting column. If None, centers the piece.
        """
        self.piece_type = piece_type
        info = PIECE_TYPES[piece_type]
        self.shape = list(info["shape"])
        self.color = info["color"]
        self.piece_id = info["id"]

        # Calculate piece width
        max_col = max(c for _, c in self.shape)
        self.width = max_col + 1

        # Starting position: top of board, centered
        self.row = 0
        if col is not None:
            self.col = col
        else:
            self.col = (BOARD_WIDTH - self.width) // 2

    def get_cells(self) -> List[Tuple[int, int]]:
        """
        Gets the absolute board positions of all cells in this piece.

        Returns:
            List[Tuple[int, int]]: A list of (row, col) tuples representing the
                                     piece's cells on the board.
        """
        return [(self.row + dr, self.col + dc) for dr, dc in self.shape]

    def to_dict(self) -> Dict[str, Any]:
        """Convert piece to a dictionary for serialization."""
        return {
            "type": self.piece_type,
            "row": self.row,
            "col": self.col,
            "shape": self.shape,
            "color": self.color,
            "id": self.piece_id,
            "width": self.width
        }

    def clone(self) -> 'Piece':
        """
        Creates a deep copy of the piece.

        Returns:
            Piece: A new Piece instance with the same attributes.
        """
        p = Piece.__new__(Piece)
        p.piece_type = self.piece_type
        p.shape = list(self.shape)
        p.color = self.color
        p.piece_id = self.piece_id
        p.width = self.width
        p.row = self.row
        p.col = self.col
        return p


class GameState:
    """
    Represents the full state of one game session.

    This class manages the game board, current and next pieces, score, and game status.
    It provides methods for moving pieces, checking for collisions, clearing lines,
    and recording game history.

    Attributes:
        width (int): The width of the game board.
        height (int): The height of the game board.
        board (List[List[int]]): The game board represented as a 2D list. 0 is empty, >0 is a piece ID.
        current_piece (Optional[Piece]): The piece currently falling.
        next_piece_type (Optional[str]): The type of the next piece to spawn.
        score (int):The current game score.
        lines_cleared (int): The total number of lines cleared.
        pieces_placed (int): The total number of pieces placed.
        game_over (bool): True if the game is over, False otherwise.
        step_count (int): The number of steps (pieces placed) taken in the game.
        seed (int): The random seed used for piece generation.
        rng (random.Random): The random number generator.
        history (List[Dict[str, Any]]): A list of game events for replay.
    """

    def __init__(self, width: int = BOARD_WIDTH, height: int = BOARD_HEIGHT, seed: Optional[int] = None):
        """
        Initializes a new game state.

        Args:
            width (int): The width of the board. Defaults to BOARD_WIDTH.
            height (int): The height of the board. Defaults to BOARD_HEIGHT.
            seed (Optional[int]): The random seed. If None, a random seed is generated.
        """
        self.width = width
        self.height = height
        self.board = [[0] * width for _ in range(height)]
        self.current_piece: Optional[Piece] = None
        self.next_piece_type: Optional[str] = None
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.step_count = 0

        # Random seed for reproducibility
        self.seed = seed if seed is not None else random.randint(0, 2**31)
        self.rng = random.Random(self.seed)

        # History for replay
        self.history: List[Dict[str, Any]] = []

        # Prepare first pieces
        self.next_piece_type = self._random_piece_type()
        self._spawn_piece()

    def add_garbage_lines(self, num_lines: int):
        """Adds garbage lines to the bottom of the board."""
        for _ in range(num_lines):
            row = [1] * self.width  # Fill with blocks (using ID 1 for simplicity)
            # Create 1-2 random holes
            num_holes = self.rng.randint(1, 2)
            hole_indices = self.rng.sample(range(self.width), num_holes)
            for col in hole_indices:
                row[col] = 0
            
            # Remove top row (if not empty, it's lost) and add new row at bottom
            self.board.pop(0)
            self.board.append(row)

    def _random_piece_type(self) -> str:
        return self.rng.choice(PIECE_NAMES)

    def _spawn_piece(self):
        """Spawn the next piece at the top of the board."""
        piece_type = self.next_piece_type
        self.next_piece_type = self._random_piece_type()
        self.current_piece = Piece(piece_type)

        # Check if spawn position is valid
        if not self._is_valid_position(self.current_piece):
            self.game_over = True
            self._record_event("game_over", {})

    def _is_valid_position(self, piece: Piece) -> bool:
        """Check if piece position is valid (within bounds and no collision)."""
        for r, c in piece.get_cells():
            if r < 0 or r >= self.height or c < 0 or c >= self.width:
                return False
            if self.board[r][c] != 0:
                return False
        return True

    def _lock_piece(self):
        """Lock the current piece into the board."""
        if self.current_piece is None:
            return
        for r, c in self.current_piece.get_cells():
            if 0 <= r < self.height and 0 <= c < self.width:
                self.board[r][c] = self.current_piece.piece_id
        self.pieces_placed += 1

    def _clear_lines(self) -> int:
        """Clear filled lines and return count of lines cleared."""
        lines_to_clear = []
        for r in range(self.height):
            if all(self.board[r][c] != 0 for c in range(self.width)):
                lines_to_clear.append(r)

        if not lines_to_clear:
            return 0

        # Remove cleared lines and add empty lines at top
        for r in sorted(lines_to_clear, reverse=True):
            del self.board[r]
        for _ in range(len(lines_to_clear)):
            self.board.insert(0, [0] * self.width)

        return len(lines_to_clear)

    def _record_event(self, event_type: str, data: Dict[str, Any]):
        """Record an event for replay."""
        self.history.append({
            "step": self.step_count,
            "event": event_type,
            "data": data,
            "board": [row[:] for row in self.board],
            "piece": {
                "type": self.current_piece.piece_type,
                "row": self.current_piece.row,
                "col": self.current_piece.col,
            } if self.current_piece else None,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "timestamp": time.time()
        })

    def move_left(self) -> bool:
        """
        Moves the current piece one column to the left.

        Returns:
            bool: True if the move was successful, False if it was blocked or the game is over.
        """
        if self.game_over or self.current_piece is None:
            return False
        self.current_piece.col -= 1
        if not self._is_valid_position(self.current_piece):
            self.current_piece.col += 1
            return False
        self._record_event("move_left", {})
        return True

    def move_right(self) -> bool:
        """
        Moves the current piece one column to the right.

        Returns:
            bool: True if the move was successful, False if it was blocked or the game is over.
        """
        if self.game_over or self.current_piece is None:
            return False
        self.current_piece.col += 1
        if not self._is_valid_position(self.current_piece):
            self.current_piece.col -= 1
            return False
        self._record_event("move_right", {})
        return True

    def move_down(self) -> bool:
        """
        Moves the current piece one row down.

        If the piece cannot move down, it is locked into place, any completed lines
        are cleared, and a new piece is spawned.

        Returns:
            bool: True if the piece moved down, False if it was locked into place or the game is over.
        """
        if self.game_over or self.current_piece is None:
            return False
        self.current_piece.row += 1
        if not self._is_valid_position(self.current_piece):
            self.current_piece.row -= 1
            # Lock piece
            self._lock_piece()
            cleared = self._clear_lines()
            if cleared > 0:
                self.lines_cleared += cleared
                # Scoring: 1 line=100, 2=300, 3=500, 4=800
                score_map = {1: 100, 2: 300, 3: 500, 4: 800}
                self.score += score_map.get(cleared, cleared * 200)
                self._record_event("lines_cleared", {"count": cleared})
            else:
                self._record_event("piece_locked", {})
            self.step_count += 1
            self._spawn_piece()
            return False
        return True

    def hard_drop(self) -> int:
        """
        Drops the current piece as far down as possible and locks it.

        Returns:
            int: The number of rows the piece was dropped.
        """
        if self.game_over or self.current_piece is None:
            return 0
        rows_dropped = 0
        while self.move_down():
            rows_dropped += 1
        # move_down returned False means piece was locked
        return rows_dropped

    def drop_and_place(self, target_col: int) -> Dict[str, Any]:
        """
        An AI-friendly method to place the current piece in a specific column.

        This method moves the piece horizontally to the `target_col` and then
        performs a hard drop.

        Args:
            target_col (int): The target column for the piece's anchor point.

        Returns:
            Dict[str, Any]: A dictionary containing the result of the operation:
                - "success" (bool): True if the piece was placed, False otherwise.
                - "rows_dropped" (int): The number of rows dropped.
                - "score_gained" (int): The score gained by placing the piece.
                - "lines_cleared" (int): The number of lines cleared.
                - "game_over" (bool): True if the game is over after this move.
                - "reason" (str): The reason for failure, if successful is False.
        """
        if self.game_over or self.current_piece is None:
            return {"success": False, "reason": "game_over" if self.game_over else "no_piece"}

        old_score = self.score
        old_lines = self.lines_cleared

        # Move piece to target column
        piece = self.current_piece
        # Calculate the leftmost column of the piece
        min_piece_col = min(dc for _, dc in piece.shape)
        actual_target = target_col - min_piece_col

        # Move horizontally
        while piece.col < actual_target:
            if not self.move_right():
                break
        while piece.col > actual_target:
            if not self.move_left():
                break

        # Hard drop
        self._record_event("drop_and_place", {"target_col": target_col})
        rows = self.hard_drop()

        return {
            "success": True,
            "rows_dropped": rows,
            "score_gained": self.score - old_score,
            "lines_cleared": self.lines_cleared - old_lines,
            "game_over": self.game_over
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the current game state.

        Returns:
            Dict[str, Any]: A dictionary containing details about the board,
                            current piece, next piece, score, and game status.
        """
        return {
            "board": [row[:] for row in self.board],
            "width": self.width,
            "height": self.height,
            "current_piece": {
                "type": self.current_piece.piece_type,
                "row": self.current_piece.row,
                "col": self.current_piece.col,
                "cells": self.current_piece.get_cells(),
                "color": self.current_piece.color,
            } if self.current_piece else None,
            "next_piece_type": self.next_piece_type,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "pieces_placed": self.pieces_placed,
            "game_over": self.game_over,
            "step_count": self.step_count,
            "seed": self.seed
        }

    def get_board_with_piece(self) -> List[List[int]]:
        """Get board with current piece rendered on it."""
        board = [row[:] for row in self.board]
        if self.current_piece:
            for r, c in self.current_piece.get_cells():
                if 0 <= r < self.height and 0 <= c < self.width:
                    board[r][c] = self.current_piece.piece_id
        return board

    def get_column_heights(self) -> List[int]:
        """Get the height of each column (from bottom)."""
        heights = []
        for c in range(self.width):
            h = 0
            for r in range(self.height):
                if self.board[r][c] != 0:
                    h = self.height - r
                    break
            heights.append(h)
        return heights

    def count_holes(self) -> int:
        """Count the number of holes (empty cells below filled cells)."""
        holes = 0
        for c in range(self.width):
            found_block = False
            for r in range(self.height):
                if self.board[r][c] != 0:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def get_bumpiness(self) -> int:
        """Sum of absolute height differences between adjacent columns."""
        heights = self.get_column_heights()
        return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

    def get_aggregate_height(self) -> int:
        """Sum of all column heights."""
        return sum(self.get_column_heights())

    def get_complete_lines_count(self) -> int:
        """Count lines that would be cleared."""
        count = 0
        for r in range(self.height):
            if all(self.board[r][c] != 0 for c in range(self.width)):
                count += 1
        return count

    def clone(self) -> 'GameState':
        """
        Creates a deep copy of the game state.

        The game history is not copied to improve performance, which is useful
        for AI search algorithms like DFS.

        Returns:
            GameState: A new GameState instance that is a copy of the current state.
        """
        gs = GameState.__new__(GameState)
        gs.width = self.width
        gs.height = self.height
        gs.board = [row[:] for row in self.board]
        gs.current_piece = self.current_piece.clone() if self.current_piece else None
        gs.next_piece_type = self.next_piece_type
        gs.score = self.score
        gs.lines_cleared = self.lines_cleared
        gs.pieces_placed = self.pieces_placed
        gs.game_over = self.game_over
        gs.step_count = self.step_count
        gs.seed = self.seed
        gs.rng = random.Random()
        gs.rng.setstate(self.rng.getstate())
        gs.history = []  # Don't copy history for performance in DFS
        return gs

    def export_history(self) -> List[Dict[str, Any]]:
        """Export full game history for replay."""
        return self.history

    def export_summary(self) -> Dict[str, Any]:
        """Export game summary statistics."""
        return {
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "pieces_placed": self.pieces_placed,
            "game_over": self.game_over,
            "step_count": self.step_count,
            "seed": self.seed,
            "final_column_heights": self.get_column_heights(),
            "final_holes": self.count_holes(),
            "final_bumpiness": self.get_bumpiness(),
            "final_aggregate_height": self.get_aggregate_height(),
        }


# ==================== Reward Functions ====================

def default_reward(state: GameState, lines_cleared: int) -> float:
    """
    Default reward function.

    Rewards lines cleared, and penalizes holes, bumpiness, and aggregate height.

    Args:
        state (GameState): The game state after placing a piece.
        lines_cleared (int): The number of lines cleared by the last move.

    Returns:
        float: The calculated reward.
    """
    reward = lines_cleared * 100.0
    reward -= state.count_holes() * 5.0
    reward -= state.get_bumpiness() * 2.0
    reward -= state.get_aggregate_height() * 0.5
    return reward


def aggressive_reward(state: GameState, lines_cleared: int) -> float:
    """
    Aggressive reward function.

    Heavily rewards multi-line clears (especially 4 lines) and penalizes
    holes and bumpiness more strictly.

    Args:
        state (GameState): The game state after placing a piece.
        lines_cleared (int): The number of lines cleared by the last move.

    Returns:
        float: The calculated reward.
    """
    line_bonus = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}
    # reward = line_bonus.get(lines_cleared, lines_cleared * 300)
    reward = 0
    reward -= state.count_holes() * 1.0
    reward -= state.get_bumpiness() * 1.0
    reward -= state.get_aggregate_height() * 1.0
    return reward


def conservative_reward(state: GameState, lines_cleared: int) -> float:
    """
    Conservative reward function.

    Focuses on keeping the board clean and low. Penalizes holes, bumpiness,
    and aggregate height, and adds an extra penalty for high columns.

    Args:
        state (GameState): The game state after placing a piece.
        lines_cleared (int): The number of lines cleared by the last move.

    Returns:
        float: The calculated reward.
    """
    reward = lines_cleared * 50.0
    reward -= state.count_holes() * 20.0
    reward -= state.get_bumpiness() * 5.0
    reward -= state.get_aggregate_height() * 2.0
    max_height = max(state.get_column_heights()) if state.get_column_heights() else 0
    if max_height > state.height * 0.6:
        reward -= (max_height - state.height * 0.6) * 10.0
    return reward


REWARD_FUNCTIONS = {
    "default": default_reward,
    "aggressive": aggressive_reward,
    "conservative": conservative_reward,
}


# ==================== DFS Algorithm ====================

class DFSResult:
    """
    Represents the result of a DFS/Beam Search.

    Stores the top-k best action sequences found during the search.

    Attributes:
        top_k (List[Tuple[float, List[Dict[str, Any]]]]): A list of tuples,
            where each tuple contains the cumulative reward and the sequence of actions.
    """

    def __init__(self):
        """Initializes a new DFSResult."""
        self.top_k: List[Tuple[float, List[Dict[str, Any]]]] = []  # (reward, action_sequence)

    def add(self, reward: float, actions: List[Dict[str, Any]], k: int):
        """
        Adds a new action sequence and its reward to the result, maintaining only the top-k.

        Args:
            reward (float): The cumulative reward for the action sequence.
            actions (List[Dict[str, Any]]): The sequence of actions.
            k (int): The maximum number of top results to keep.
        """
        self.top_k.append((reward, actions))
        self.top_k.sort(key=lambda x: x[0], reverse=True)
        if len(self.top_k) > k:
            self.top_k = self.top_k[:k]

    def to_dict(self) -> List[Dict[str, Any]]:
        return [
            {"rank": i + 1, "reward": r, "actions": a}
            for i, (r, a) in enumerate(self.top_k)
        ]


def get_possible_placements(state: GameState) -> List[int]:
    """
    Returns a list of all valid target columns for the current piece in the given state.

    A target column is valid if the piece can be placed there without colliding
    with existing blocks or going out of bounds.

    Args:
        state (GameState): The current game state.

    Returns:
        List[int]: A list of valid column indices for the piece's anchor point.
    """
    if state.current_piece is None:
        return []

    valid_cols = []
    piece = state.current_piece
    min_dc = min(dc for _, dc in piece.shape)
    max_dc = max(dc for _, dc in piece.shape)

    for target_col in range(0 - min_dc, state.width - max_dc):
        # Test if placement is valid
        test_state = state.clone()
        test_piece = test_state.current_piece
        if test_piece is None:
            continue
        # Move to target
        actual_target = target_col - min_dc
        test_piece.col = actual_target
        if test_state._is_valid_position(test_piece):
            valid_cols.append(target_col)

    return valid_cols


def dfs_search(
    state: GameState,
    reward_fn: Callable[[GameState, int], float],
    depth_limit: int = 3,
    top_k: int = 5,
    beam_width: int = 10,
) -> DFSResult:
    """
    Performs a Depth-First Search (DFS) or Beam Search to find the best sequences of moves.

    This function explores possible moves up to a specified depth, using a reward function
    to evaluate the resulting states. It uses a beam search approach to keep the search
    manageable by expanding only the most promising paths at each depth.

    Args:
        state (GameState): The initial game state to start the search from.
        reward_fn (Callable[[GameState, int], float]): A function that calculates the reward for a given state and lines cleared.
        depth_limit (int): The maximum number of moves to look ahead. Defaults to 3.
        top_k (int): The number of best action sequences to return. Defaults to 5.
        beam_width (int): The maximum number of paths to keep at each depth. Defaults to 10.

    Returns:
        DFSResult: An object containing the top-k action sequences and their cumulative rewards.
                   Each action in the sequence includes the resulting state.
    """
    result = DFSResult()

    # The beam stores tuples of: (cumulative_reward, current_state, action_history)
    # Start with the initial state and zero reward
    current_beam = [(0.0, state, [])]

    for depth in range(depth_limit):
        candidates = []
        
        # Expand all states in the current beam
        for cum_reward, curr_state, act_history in current_beam:
            if curr_state.game_over or curr_state.current_piece is None:
                continue

            placements = get_possible_placements(curr_state)
            piece_type = curr_state.current_piece.piece_type

            for col in placements:
                # Clone state and simulate placement
                sim_state = curr_state.clone()
                old_lines = sim_state.lines_cleared
                sim_result = sim_state.drop_and_place(col)

                if sim_result["success"]:
                    new_lines = sim_state.lines_cleared - old_lines
                    reward = reward_fn(sim_state, new_lines)
                    new_cum_reward = cum_reward + reward

                    # Create the action dict, now including the resulting state
                    action_dict = {
                        "piece_type": piece_type,
                        "target_col": col,
                        "lines_cleared": new_lines,
                        "reward": reward,
                        "resulting_state": sim_state.get_state_dict() # Add full state
                    }
                    new_act_history = act_history + [action_dict]

                    if sim_state.game_over:
                        # Penalize game over and add to candidates, but it won't be expanded further
                        candidates.append((new_cum_reward - 1000.0, sim_state, new_act_history))
                    else:
                        candidates.append((new_cum_reward, sim_state, new_act_history))

        # Sort all candidates from this depth by cumulative reward (descending)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Select the top 'beam_width' candidates to form the next beam
        current_beam = candidates[:beam_width]
        
        # If the beam is empty, we can't go any deeper
        if not current_beam:
            break

    # Add the paths from the final depth to the result
    for r, s, acts in current_beam:
        if acts: # Only add if there's at least one action
            result.add(r, acts, top_k)

    return result


# ==================== Simple AI Agents ====================

class GreedyAgent:
    """
    An AI agent that uses a greedy strategy.

    It evaluates all possible placements for the current piece and chooses the one
    that yields the highest immediate reward according to the given reward function.
    """

    def __init__(self, reward_fn: Callable = default_reward):
        """
        Initializes a new GreedyAgent.

        Args:
            reward_fn (Callable): The reward function to use for evaluation.
                                  Defaults to `default_reward`.
        """
        self.reward_fn = reward_fn
        self.name = "GreedyAgent"

    def choose_action(self, state: GameState) -> Optional[int]:
        """
        Chooses the best target column for the current piece in the given state.

        Args:
            state (GameState): The current game state.

        Returns:
            Optional[int]: The best target column, or None if the game is over or no piece is falling.
        """
        if state.game_over or state.current_piece is None:
            return None

        best_col = None
        best_reward = float('-inf')

        placements = get_possible_placements(state)
        for col in placements:
            sim_state = state.clone()
            old_lines = sim_state.lines_cleared
            sim_state.drop_and_place(col)
            new_lines = sim_state.lines_cleared - old_lines
            reward = self.reward_fn(sim_state, new_lines)
            if reward > best_reward:
                best_reward = reward
                best_col = col

        return best_col


class DFSAgent:
    """
    An AI agent that uses Depth-First Search (DFS) / Beam Search.

    It uses the `dfs_search` function to find the best sequence of moves up to a
    specified depth and chooses the first action from the best sequence.
    """

    def __init__(self, reward_fn: Callable = default_reward, depth: int = 3, top_k: int = 5, beam_width: int = 10):
        """
        Initializes a new DFSAgent.

        Args:
            reward_fn (Callable): The reward function to use for evaluation.
                                  Defaults to `default_reward`.
            depth (int): The search depth. Defaults to 3.
            top_k (int): The number of top results to keep in the search. Defaults to 5.
            beam_width (int): The beam width for the search. Defaults to 10.
        """
        self.reward_fn = reward_fn
        self.depth = depth
        self.top_k = top_k
        self.beam_width = beam_width
        self.name = f"DFSAgent(depth={depth}, beam={beam_width})"

    def choose_action(self, state: GameState) -> Optional[int]:
        """
        Chooses the best target column for the current piece using DFS/Beam Search.

        Args:
            state (GameState): The current game state.

        Returns:
            Optional[int]: The best target column, or None if the game is over or no piece is falling.
        """
        if state.game_over or state.current_piece is None:
            return None

        # Use the dfs_search function
        result = dfs_search(state, self.reward_fn, self.depth, self.top_k, self.beam_width)
        
        # result.top_k is a list of (reward, action_history) tuples.
        # We take the best one (index 0), then its action history (index 1),
        # then the first action in that history (index 0), and finally its target column.
        if result.top_k and result.top_k[0][1]:
            return result.top_k[0][1][0]["target_col"]
        return None


class BeamSearchAgent:
    """
    An AI agent that uses Beam Search.

    This is functionally identical to `DFSAgent` but with a different name for clarity.
    It uses the `dfs_search` function (which implements beam search) to find the best
    sequence of moves.
    """

    def __init__(self, reward_fn: Callable = default_reward, depth: int = 3, beam_width: int = 10, top_k: int = 3):
        """
        Initializes a new BeamSearchAgent.

        Args:
            reward_fn (Callable): The reward function to use for evaluation.
                                  Defaults to `default_reward`.
            depth (int): The search depth. Defaults to 3.
            beam_width (int): The beam width for the search. Defaults to 10.
            top_k (int): The number of top results to keep in the search. Defaults to 3.
        """
        self.reward_fn = reward_fn
        self.depth = depth
        self.beam_width = beam_width
        self.top_k = top_k
        self.name = f"BeamSearchAgent(depth={depth}, beam={beam_width})"

    def choose_action(self, state: GameState) -> Optional[int]:
        """
        Chooses the best target column for the current piece using Beam Search.

        Args:
            state (GameState): The current game state.

        Returns:
            Optional[int]: The best target column, or None if the game is over or no piece is falling.
        """
        if state.game_over or state.current_piece is None:
            return None

        # Use the new dfs_search function
        result = dfs_search(state, self.reward_fn, self.depth, self.top_k, self.beam_width)
        
        # result.top_k is a list of (reward, action_history) tuples.
        # We take the best one (index 0), then its action history (index 1),
        # then the first action in that history (index 0), and finally its target column.
        if result.top_k and result.top_k[0][1]:
            return result.top_k[0][1][0]["target_col"]
        return None


class RandomAgent:
    """
    An AI agent that plays randomly.

    It chooses a random valid placement for the current piece.
    """

    def __init__(self):
        """Initializes a new RandomAgent."""
        self.name = "RandomAgent"

    def choose_action(self, state: GameState) -> Optional[int]:
        """
        Chooses a random target column for the current piece.

        Args:
            state (GameState): The current game state.

        Returns:
            Optional[int]: A random valid target column, or None if the game is over or no piece is falling.
        """
        if state.game_over or state.current_piece is None:
            return None
        placements = get_possible_placements(state)
        if placements:
            return random.choice(placements)
        return None
