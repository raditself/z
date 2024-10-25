
import chess
import chess.syzygy
from functools import lru_cache

class EndgameTablebase:
    def __init__(self):
        tablebase_path = "/home/user/z/data/syzygy/shakmaty-syzygy/3-4-5"
        self.tablebase = chess.syzygy.open_tablebase(tablebase_path)

    @lru_cache(maxsize=10000)
    def probe_wdl(self, board_fen):
        """
        Probe the Win-Draw-Loss (WDL) table.
        Returns:
        2: Win
        0: Draw
        -2: Loss
        None: Position not in tablebase
        """
        board = chess.Board(board_fen)
        try:
            return self.tablebase.probe_wdl(board)
        except ValueError:
            return None

    @lru_cache(maxsize=10000)
    def probe_dtz(self, board_fen):
        """
        Probe the Distance-To-Zero (DTZ) table.
        Returns the number of moves to reach a zeroing position (capture or pawn move).
        None if position not in tablebase.
        """
        board = chess.Board(board_fen)
        try:
            return self.tablebase.probe_dtz(board)
        except ValueError:
            return None

    def get_best_move(self, board):
        """
        Get the best move according to the tablebase.
        Returns the best move or None if not in tablebase.
        """
        best_move = None
        best_wdl = -3  # Worse than any real WDL value

        for move in board.legal_moves:
            board.push(move)
            wdl = self.probe_wdl(board.fen())
            board.pop()

            if wdl is not None:
                wdl = -wdl  # Negate because we're looking from the opponent's perspective
                if wdl > best_wdl:
                    best_wdl = wdl
                    best_move = move

        return best_move
