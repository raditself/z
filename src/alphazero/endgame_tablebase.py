
import chess
import random

class EndgameTablebase:
    def __init__(self):
        pass

    def probe_wdl(self, board):
        """
        Simulate probing the Win-Draw-Loss (WDL) table.
        Returns:
        2: Win
        0: Draw
        -2: Loss
        """
        # Simplified logic: If the side to move has more material, it's winning
        if board.turn == chess.WHITE:
            return 2 if self._white_material(board) > self._black_material(board) else -2
        else:
            return 2 if self._black_material(board) > self._white_material(board) else -2

    def probe_dtz(self, board):
        """
        Simulate probing the Distance-To-Zero (DTZ) table.
        Returns a random number of moves to reach a zeroing position.
        """
        return random.randint(1, 50)

    def get_best_move(self, board):
        """
        Get a 'best' move based on simple heuristics.
        """
        best_move = None
        best_score = float('-inf')

        for move in board.legal_moves:
            board.push(move)
            score = self._evaluate_position(board)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _evaluate_position(self, board):
        """
        Simple position evaluation.
        """
        return self._white_material(board) - self._black_material(board)

    def _white_material(self, board):
        return (len(board.pieces(chess.PAWN, chess.WHITE)) +
                3 * len(board.pieces(chess.KNIGHT, chess.WHITE)) +
                3 * len(board.pieces(chess.BISHOP, chess.WHITE)) +
                5 * len(board.pieces(chess.ROOK, chess.WHITE)) +
                9 * len(board.pieces(chess.QUEEN, chess.WHITE)))

    def _black_material(self, board):
        return (len(board.pieces(chess.PAWN, chess.BLACK)) +
                3 * len(board.pieces(chess.KNIGHT, chess.BLACK)) +
                3 * len(board.pieces(chess.BISHOP, chess.BLACK)) +
                5 * len(board.pieces(chess.ROOK, chess.BLACK)) +
                9 * len(board.pieces(chess.QUEEN, chess.BLACK)))
