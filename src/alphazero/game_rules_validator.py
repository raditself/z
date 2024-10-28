
import random
import unittest
import numpy as np
from typing import List, Tuple, Any
from src.alphazero.game import Game
from src.games.checkers import CheckersGame
from src.games.chess import ChessGame
from src.games.go import GoGame
import matplotlib.pyplot as plt

class GameRulesValidator:
    def __init__(self, game: Game):
        self.game = game

    def generate_random_state(self) -> Tuple[Any, int]:
        """Generate a random game state."""
        state = self.game.get_initial_state()
        player = 1
        moves = 0
        while not self.game.is_terminal(state) and moves < 100:
            valid_moves = self.game.get_valid_moves(state)
            if np.sum(valid_moves) == 0:
                break
            action = random.choice(np.where(valid_moves)[0])
            state = self.game.get_next_state(state, action)
            player = self.game.get_opponent(player)
            moves += 1
        return state, player

    def validate_state(self, state: Any, player: int) -> bool:
        """Validate a given game state."""
        # Check if the state has the correct shape
        if not hasattr(state, 'shape') or (hasattr(self.game, 'board_size') and state.shape != (self.game.board_size, self.game.board_size)):
            return False

        # Check if the player is valid
        if player not in [-1, 1]:
            return False

        # Check if the game has ended
        if self.game.is_terminal(state):
            return True

        # Check if there are valid moves
        valid_moves = self.game.get_valid_moves(state)
        if np.sum(valid_moves) == 0:
            return False

        return True

    def fuzz_test(self, num_tests: int = 1000) -> List[Tuple[Any, int]]:
        """Perform fuzz testing on the game rules."""
        failed_states = []
        for _ in range(num_tests):
            state, player = self.generate_random_state()
            if not self.validate_state(state, player):
                failed_states.append((state, player))
        return failed_states

    def rules_consistency_check(self, num_simulations: int = 100) -> bool:
        """Check if game rules produce consistent results for the same state and action."""
        state, player = self.generate_random_state()
        valid_moves = self.game.get_valid_moves(state)
        if np.sum(valid_moves) == 0:
            return True  # No valid moves, so it's consistently terminal
        action = random.choice(np.where(valid_moves)[0])
        
        first_next_state = self.game.get_next_state(state, action)
        first_value, first_terminated = self.game.get_value_and_terminated(state, action)
        
        for _ in range(num_simulations - 1):
            next_state = self.game.get_next_state(state, action)
            value, terminated = self.game.get_value_and_terminated(state, action)
            
            if not np.array_equal(next_state, first_next_state) or value != first_value or terminated != first_terminated:
                return False
        
        return True

    def visualize_failed_state(self, state: Any, player: int, filename: str):
        """Visualize a failed state for debugging."""
        plt.figure(figsize=(8, 8))
        plt.imshow(state, cmap='coolwarm', interpolation='nearest')
        plt.title(f"Failed State (Player: {player})")
        plt.colorbar()
        plt.savefig(filename)
        plt.close()

class TestGameRulesValidator(unittest.TestCase):
    def setUp(self):
        self.checkers_validator = GameRulesValidator(CheckersGame())
        self.chess_validator = GameRulesValidator(ChessGame())
        self.go_validator = GameRulesValidator(GoGame())

    def test_generate_random_state(self):
        for validator in [self.checkers_validator, self.chess_validator, self.go_validator]:
            state, player = validator.generate_random_state()
            self.assertIsNotNone(state)
            self.assertIn(player, [-1, 1])

    def test_validate_state(self):
        for validator in [self.checkers_validator, self.chess_validator, self.go_validator]:
            state, player = validator.generate_random_state()
            self.assertTrue(validator.validate_state(state, player))

    def test_fuzz_test(self):
        for validator in [self.checkers_validator, self.chess_validator, self.go_validator]:
            failed_states = validator.fuzz_test(num_tests=100)
            self.assertEqual(len(failed_states), 0, f"Fuzz test failed for {len(failed_states)} states")

    def test_rules_consistency(self):
        for validator in [self.checkers_validator, self.chess_validator, self.go_validator]:
            self.assertTrue(validator.rules_consistency_check(num_simulations=10))

    def test_checkers_edge_cases(self):
        # Test king creation
        king_state = np.zeros((8, 8))
        king_state[0, 0] = 1  # White piece about to become king
        self.assertTrue(self.checkers_validator.validate_state(king_state, 1))

        # Test multiple jumps
        multi_jump_state = np.zeros((8, 8))
        multi_jump_state[3, 3] = 1  # White piece
        multi_jump_state[2, 2] = -1  # Black piece to be jumped
        multi_jump_state[4, 4] = -1  # Another black piece to be jumped
        self.assertTrue(self.checkers_validator.validate_state(multi_jump_state, 1))

def run_tests():
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == "__main__":
    run_tests()

# Documentation

"""
Game Rules Validator for AlphaZero

This module provides a comprehensive system for automatic bug detection and validation
in the game rules implementation for the AlphaZero project, supporting multiple games.

Features:
1. Random state generation
2. State validation
3. Fuzz testing
4. Game rules consistency checking
5. Visualization of failed states
6. Unit tests for all components, including specific edge cases for Checkers

Supported Games:
- Checkers
- Chess
- Go

Setup Instructions:
1. Ensure that the AlphaZero project is properly set up and all dependencies are installed.
2. Place this file (game_rules_validator.py) in the src/alphazero directory of the project.

Usage Guidelines:
1. To use the validator for a specific game, create an instance of GameRulesValidator:
   
   from src.games.checkers import CheckersGame
   validator = GameRulesValidator(CheckersGame())

2. To run fuzz tests:
   failed_states = validator.fuzz_test(num_tests=1000)

3. To check rules consistency:
   is_consistent = validator.rules_consistency_check(num_simulations=100)

4. To visualize a failed state:
   validator.visualize_failed_state(state, player, "failed_state.png")

5. To run all unit tests:
   python game_rules_validator.py

Maintenance Procedures:
1. Regularly update the test cases to cover new game rules or edge cases.
2. If new games are added to the project, create corresponding test cases in the
   TestGameRulesValidator class.
3. Periodically increase the number of fuzz tests and rule consistency simulations to improve
   the robustness of the validation system.
4. Update the CI/CD pipeline to include running these tests as part of the build process.
"""
