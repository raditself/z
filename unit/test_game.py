
import unittest
from src.alphazero.game import Game
from src.alphazero.game_state import GameState

class TestGame(unittest.TestCase):
    def setUp(self):
        self.game = Game()

    def test_initial_state(self):
        state = self.game.get_initial_state()
        self.assertIsInstance(state, GameState)
        self.assertEqual(state.to_play, 1)  # Assuming white starts

    def test_get_valid_moves(self):
        state = self.game.get_initial_state()
        valid_moves = self.game.get_valid_moves(state)
        self.assertIsInstance(valid_moves, list)
        self.assertTrue(len(valid_moves) > 0)  # There should be valid moves in the initial state

    def test_is_game_over(self):
        state = self.game.get_initial_state()
        self.assertFalse(self.game.is_game_over(state))

if __name__ == '__main__':
    unittest.main()
