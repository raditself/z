
import unittest
from src.games.chess import ChessGame

class TestChessGame(unittest.TestCase):
    def setUp(self):
        self.game = ChessGame()

    def test_initial_board(self):
        board = self.game.get_initial_board()
        self.assertEqual(board.fen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def test_valid_moves(self):
        board = self.game.get_initial_board()
        valid_moves = self.game.get_valid_moves(board)
        self.assertEqual(len(valid_moves), 20)  # 16 pawn moves + 4 knight moves

    def test_make_move(self):
        board = self.game.get_initial_board()
        new_board = self.game.make_move(board, "e2e4")
        self.assertEqual(new_board.fen(), "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")

    def test_is_game_over(self):
        board = self.game.get_initial_board()
        self.assertFalse(self.game.is_game_over(board))

        # Set up a checkmate position
        board.set_fen("rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        self.assertTrue(self.game.is_game_over(board))

    def test_get_winner(self):
        board = self.game.get_initial_board()
        self.assertIsNone(self.game.get_winner(board))

        # Set up a checkmate position for black
        board.set_fen("rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        self.assertEqual(self.game.get_winner(board), -1)  # Black wins

    def test_get_current_player(self):
        board = self.game.get_initial_board()
        self.assertEqual(self.game.get_current_player(board), 1)  # White to move

        new_board = self.game.make_move(board, "e2e4")
        self.assertEqual(self.game.get_current_player(new_board), -1)  # Black to move

if __name__ == '__main__':
    unittest.main()
