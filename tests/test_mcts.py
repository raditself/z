
import unittest
import numpy as np
from src.alphazero.mcts_nn import MCTS, MCTSNode
from src.games.chess import ChessGame
from src.alphazero.model import ChessModel

class MockModel:
    def predict(self, state):
        # Mock prediction for testing purposes
        return np.ones(64) / 64, 0.5

class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.game = ChessGame()
        self.model = MockModel()
        self.mcts = MCTS(self.game, self.model, num_simulations=10)

    def test_search(self):
        initial_state = self.game.get_initial_board()
        action_probs = self.mcts.search(initial_state)
        
        self.assertEqual(len(action_probs), 64)  # 64 possible moves in chess
        self.assertAlmostEqual(sum(action_probs), 1.0, places=6)

    def test_select_child(self):
        root = MCTSNode(self.game.get_initial_board())
        child1 = MCTSNode(self.game.get_initial_board(), parent=root, action=0)
        child2 = MCTSNode(self.game.get_initial_board(), parent=root, action=1)
        root.children = [child1, child2]

        child1.visits = 10
        child1.value = 5
        child1.prior = 0.6

        child2.visits = 5
        child2.value = 3
        child2.prior = 0.4

        selected_child = self.mcts.select_child(root)
        self.assertEqual(selected_child, child2)  # Child2 should be selected due to exploration bonus

    def test_expand_and_evaluate(self):
        node = MCTSNode(self.game.get_initial_board())
        value = self.mcts.expand_and_evaluate(node)

        self.assertIsInstance(value, float)
        self.assertGreater(len(node.children), 0)

    def test_backpropagate(self):
        root = MCTSNode(self.game.get_initial_board())
        child = MCTSNode(self.game.get_initial_board(), parent=root, action=0)
        grandchild = MCTSNode(self.game.get_initial_board(), parent=child, action=1)

        search_path = [root, child, grandchild]
        self.mcts.backpropagate(search_path, 1.0)

        self.assertEqual(root.visits, 1)
        self.assertEqual(root.value, 1.0)
        self.assertEqual(child.visits, 1)
        self.assertEqual(child.value, -1.0)
        self.assertEqual(grandchild.visits, 1)
        self.assertEqual(grandchild.value, 1.0)

if __name__ == '__main__':
    unittest.main()
