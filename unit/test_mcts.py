
import unittest
from src.alphazero.mcts import MCTS
from src.alphazero.game import Game

class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.game = Game()
        self.mcts = MCTS(self.game)

    def test_search(self):
        state = self.game.get_initial_state()
        action_probs = self.mcts.search(state)
        
        self.assertIsInstance(action_probs, list)
        self.assertEqual(len(action_probs), self.game.action_size)
        self.assertAlmostEqual(sum(action_probs), 1.0, places=6)

if __name__ == '__main__':
    unittest.main()
