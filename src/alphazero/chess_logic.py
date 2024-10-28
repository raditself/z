import chess

class ChessGame:
    def __init__(self, initial_time=600, increment=10, variant='standard'):
        self.board = chess.Board()
        self.timer = None  # We'll implement this later if needed

    def make_move(self, move):
        self.board.push(move)

    def get_piece_at(self, square):
        piece = self.board.piece_at(square)
        return piece.symbol() if piece else '.'

    def __str__(self):
        return self.board.fen()

    def get_fen(self):
        return self.board.fen()

