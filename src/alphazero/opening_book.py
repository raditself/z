import random
import chess
import chess.pgn
import io

class OpeningBook:
    def __init__(self):
        self.openings = {}

    def add_opening(self, moves):
        board = chess.Board()
        key = board.fen()
        for move in moves:
            if key not in self.openings:
                self.openings[key] = []
            self.openings[key].append(move)
            board.push_san(move)
            key = board.fen()

    def get_move(self, board):
        key = board.fen()
        if key in self.openings:
            return random.choice(self.openings[key])
        return None

    def load_from_pgn(self, pgn_file):
        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                moves = [move.san() for move in game.mainline_moves()]
                self.add_opening(moves[:10])  # Add first 10 moves of each game

    def load_from_eco(self, eco_file):
        with open(eco_file, 'r') as f:
            for line in f:
                parts = line.strip().split('"')
                if len(parts) >= 4:
                    moves = parts[3].split()
                    self.add_opening(moves)

def create_sample_opening_book():
    book = OpeningBook()
    
    # Add some common openings
    book.add_opening(["e4", "e5", "Nf3", "Nc6", "Bb5"])  # Ruy Lopez
    book.add_opening(["e4", "e5", "Nf3", "Nc6", "Bc4"])  # Italian Game
    book.add_opening(["d4", "d5", "c4"])  # Queen's Gambit
    book.add_opening(["e4", "c5"])  # Sicilian Defense
    book.add_opening(["e4", "e6"])  # French Defense
    
    return book

# Usage example
if __name__ == "__main__":
    book = create_sample_opening_book()
    
    # Test the opening book
    board = chess.Board()
    for _ in range(3):
        move = book.get_move(board)
        if move:
            print(f"Suggested move: {move}")
            board.push_san(move)
        else:
            print("No move found in opening book")
            break
    
    print("Final position:")
    print(board)
