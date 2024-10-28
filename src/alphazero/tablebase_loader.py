
import chess
import chess.syzygy
import os

class TablebaseLoader:
    def __init__(self, tablebase_path):
        self.tablebase_path = tablebase_path
        self.tablebase = None
        self.loaded_pieces = 0

    def load_tablebase(self, num_pieces):
        if num_pieces <= self.loaded_pieces:
            return

        try:
            self.tablebase = chess.syzygy.open_tablebase(self.tablebase_path, load_wdl=True, load_dtz=True)
            self.loaded_pieces = num_pieces
            print(f"Loaded {num_pieces}-piece tablebase")
        except Exception as e:
            print(f"Error loading {num_pieces}-piece tablebase: {str(e)}")

    def probe_wdl(self, board):
        if self.tablebase is None:
            return None

        try:
            return self.tablebase.probe_wdl(board)
        except ValueError:
            return None

    def probe_dtz(self, board):
        if self.tablebase is None:
            return None

        try:
            return self.tablebase.probe_dtz(board)
        except ValueError:
            return None

    def get_best_move(self, board):
        if self.tablebase is None:
            return None

        best_move = None
        best_wdl = -2

        for move in board.legal_moves:
            board.push(move)
            wdl = -self.probe_wdl(board)
            board.pop()

            if wdl is not None and wdl > best_wdl:
                best_wdl = wdl
                best_move = move

        return best_move

def progressive_tablebase_loading(board, loader):
    piece_count = chess.popcount(board.occupied)
    
    if piece_count <= 7 and loader.loaded_pieces < 7:
        loader.load_tablebase(7)
    elif piece_count <= 6 and loader.loaded_pieces < 6:
        loader.load_tablebase(6)
    elif piece_count <= 5 and loader.loaded_pieces < 5:
        loader.load_tablebase(5)
    elif piece_count <= 4 and loader.loaded_pieces < 4:
        loader.load_tablebase(4)
    elif piece_count <= 3 and loader.loaded_pieces < 3:
        loader.load_tablebase(3)

    return loader.get_best_move(board)
