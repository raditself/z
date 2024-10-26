
import unittest
import chess
import chess.pgn
import io
from src.alphazero.game_analysis import GameAnalysis

class MockChessAI:
    def get_move(self, board, temperature=1.0):
        return 0.5  # Mock evaluation

class TestGameAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = GameAnalysis('mock_model_path')
        self.analyzer.ai = MockChessAI()

    def test_load_pgn(self):
        pgn_string = """
[Event "Test Game"]
[Site "Python unittest"]
[Date "2023.05.10"]
[Round "1"]
[White "Player 1"]
[Black "Player 2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0
"""
        pgn_file = io.StringIO(pgn_string)
        game = self.analyzer.load_pgn(pgn_file)
        
        self.assertIsInstance(game, chess.pgn.Game)
        self.assertEqual(game.headers["White"], "Player 1")
        self.assertEqual(game.headers["Black"], "Player 2")

    def test_analyze_game(self):
        pgn_string = "1. e4 e5 2. Nf3 Nc6 3. Bb5 1-0"
        pgn_file = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_file)
        
        analysis = self.analyzer.analyze_game(game, depth=3)
        
        self.assertEqual(len(analysis), 3)
        for move_analysis in analysis:
            self.assertIn('move', move_analysis)
            self.assertIn('fen', move_analysis)
            self.assertIn('evaluation', move_analysis)

    def test_find_blunders(self):
        analysis = [
            {'move': 'e4', 'evaluation': 0.1},
            {'move': 'e5', 'evaluation': 0.2},
            {'move': 'Nf3', 'evaluation': 0.1},
            {'move': 'Nc6', 'evaluation': -2.0},  # Blunder
            {'move': 'Bb5', 'evaluation': 2.2},
        ]
        
        blunders = self.analyzer.find_blunders(analysis, threshold=1.5)
        
        self.assertEqual(len(blunders), 1)
        self.assertEqual(blunders[0]['move'], 'Nc6')
        self.assertAlmostEqual(blunders[0]['prev_eval'], 0.1)
        self.assertAlmostEqual(blunders[0]['curr_eval'], -2.0)

if __name__ == '__main__':
    unittest.main()
