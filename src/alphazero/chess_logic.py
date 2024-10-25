
import chess
import time

class ChessGame:
    def __init__(self, initial_time=600):  # 10 minutes by default
        self.board = chess.Board()
        self.white_time = initial_time
        self.black_time = initial_time
        self.last_move_time = time.time()

    def is_valid_move(self, from_square, to_square):
        move = chess.Move(from_square, to_square)
        return move in self.board.legal_moves

    def make_move(self, from_square, to_square):
        move = chess.Move(from_square, to_square)
        if move in self.board.legal_moves:
            self.update_clock()
            self.board.push(move)
            self.last_move_time = time.time()
            return True
        return False

    def update_clock(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_move_time
        if self.board.turn == chess.WHITE:
            self.white_time -= elapsed_time
        else:
            self.black_time -= elapsed_time

    def get_piece_at(self, square):
        piece = self.board.piece_at(square)
        return piece.symbol() if piece else '.'

    def is_game_over(self):
        return self.board.is_game_over() or self.white_time <= 0 or self.black_time <= 0

    def get_result(self):
        if self.white_time <= 0:
            return "0-1"
        elif self.black_time <= 0:
            return "1-0"
        return self.board.result()

    def get_fen(self):
        return self.board.fen()

    def set_fen(self, fen):
        self.board.set_fen(fen)

    def get_legal_moves(self):
        return list(self.board.legal_moves)

    def get_current_player_time(self):
        return self.white_time if self.board.turn == chess.WHITE else self.black_time

    def format_time(self, seconds):
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"
