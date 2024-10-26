
import unittest
from unittest.mock import MagicMock, patch
from src.alphazero.model_evaluator import ModelEvaluator
from src.alphazero.game import Game
from src.alphazero.alphazero import AlphaZero

class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        self.game = MagicMock(spec=Game)
        self.evaluator = ModelEvaluator(self.game, num_games=10)

    def test_add_model(self):
        model = MagicMock(spec=AlphaZero)
        self.evaluator.add_model("model1", model)
        self.assertIn("model1", self.evaluator.models)
        self.assertIn("model1", self.evaluator.rating_system.players)

    def test_evaluate_model(self):
        model1 = MagicMock(spec=AlphaZero)
        model2 = MagicMock(spec=AlphaZero)
        self.evaluator.add_model("model1", model1)
        self.evaluator.add_model("model2", model2)

        # Mock the _play_game method to always return 1 (win for model1)
        self.evaluator._play_game = MagicMock(return_value=1)

        score = self.evaluator.evaluate_model("model1", "model2")
        self.assertEqual(score, 1.0)
        self.assertEqual(self.evaluator._play_game.call_count, 10)

    def test_get_model_rating(self):
        model = MagicMock(spec=AlphaZero)
        self.evaluator.add_model("model1", model)
        rating = self.evaluator.get_model_rating("model1")
        self.assertIsInstance(rating, float)

    def test_get_top_models(self):
        for i in range(5):
            model = MagicMock(spec=AlphaZero)
            self.evaluator.add_model(f"model{i}", model)
            self.evaluator.rating_system.players[f"model{i}"].rating = 1500 + i * 100

        top_models = self.evaluator.get_top_models(3)
        self.assertEqual(len(top_models), 3)
        self.assertEqual(top_models[0][0], "model4")
        self.assertEqual(top_models[1][0], "model3")
        self.assertEqual(top_models[2][0], "model2")

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="model1,1600\nmodel2,1550\n")
    def test_load_ratings(self, mock_open):
        self.evaluator.load_ratings("dummy_file.csv")
        self.assertIn("model1", self.evaluator.rating_system.players)
        self.assertIn("model2", self.evaluator.rating_system.players)
        self.assertEqual(self.evaluator.rating_system.get_rating("model1"), 1600)
        self.assertEqual(self.evaluator.rating_system.get_rating("model2"), 1550)

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_save_ratings(self, mock_open):
        model1 = MagicMock(spec=AlphaZero)
        model2 = MagicMock(spec=AlphaZero)
        self.evaluator.add_model("model1", model1)
        self.evaluator.add_model("model2", model2)
        self.evaluator.rating_system.players["model1"].rating = 1600
        self.evaluator.rating_system.players["model2"].rating = 1550

        self.evaluator.save_ratings("dummy_file.csv")
        mock_open.assert_called_once_with("dummy_file.csv", 'w')
        handle = mock_open()
        handle.write.assert_any_call("model1,1600.0\n")
        handle.write.assert_any_call("model2,1550.0\n")

if __name__ == '__main__':
    unittest.main()
