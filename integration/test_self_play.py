
import unittest
from src.alphazero.self_play import execute_self_play
from src.alphazero.game import Game
from src.alphazero.model import AlphaZeroNetwork

class TestSelfPlay(unittest.TestCase):
    def setUp(self):
        self.game = Game()
        input_shape = (3, 8, 8)  # Example input shape
        action_size = 4672  # Example action size for chess
        self.model = AlphaZeroNetwork(input_shape, action_size)

    def test_execute_self_play(self):
        num_games = 2
        temperature = 1.0
        examples = execute_self_play(self.model, self.game, num_games, temperature)
        
        self.assertIsInstance(examples, list)
        self.assertEqual(len(examples), num_games)
        
        for example in examples:
            self.assertIsInstance(example, tuple)
            self.assertEqual(len(example), 3)  # (state, policy, value)

if __name__ == '__main__':
    unittest.main()
