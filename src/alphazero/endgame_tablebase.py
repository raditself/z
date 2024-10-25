
import chess
import chess.syzygy
from functools import lru_cache
import os

class EndgameTablebase:
    def __init__(self):
        tablebase_path = "/home/user/z/data/syzygy"
        self.tablebase = chess.syzygy.Tablebase()
        
        # Load all available tablebase files
        for root, dirs, files in os.walk(tablebase_path):
            for file in files:
                if file.endswith('.rtbw') or file.endswith('.rtbz'):
                    self.tablebase.add_directory(root)
                    break  # Only need to add each directory once

    @lru_cache(maxsize=100000)
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
        except (chess.syzygy.MissingTableError, ValueError):
            return None

    @lru_cache(maxsize=100000)
    def probe_dtz(self, board_fen):
        """
        Probe the Distance-To-Zero (DTZ) table.
        Returns the number of moves to reach a zeroing position (capture or pawn move).
        None if position not in tablebase.
        """
        board = chess.Board(board_fen)
        try:
            return self.tablebase.probe_dtz(board)
        except (chess.syzygy.MissingTableError, ValueError):
            return None

    def get_best_move(self, board):
        """
        Get the best move according to the tablebase.
        Returns the best move or None if not in tablebase.
        """
        best_move = None
        best_wdl = -3  # Worse than any real WDL value
        best_dtz = float('inf')

        for move in board.legal_moves:
            board.push(move)
            wdl = self.probe_wdl(board.fen())
            dtz = self.probe_dtz(board.fen())
            board.pop()

            if wdl is not None:
                wdl = -wdl  # Negate because we're looking from the opponent's perspective
                if wdl > best_wdl or (wdl == best_wdl and dtz < best_dtz):
                    best_wdl = wdl
                    best_dtz = dtz
                    best_move = move

        return best_move

    def should_use_tablebase(self, board, time_left):
        """
        Decide whether to use the tablebase based on position complexity and time left.
        """
        piece_count = sum(1 for _ in board.pieces())
        
        # Always use tablebase for 7 pieces or fewer
        if piece_count <= 7:
            return True
        
        # Use tablebase for 8 pieces if we have enough time
        if piece_count == 8 and time_left > 30:  # 30 seconds threshold
            return True
        
        return False
