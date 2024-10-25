
import random
import chess
import chess.pgn
import chess.variant
import io

class OpeningBook:
    def __init__(self):
        self.openings = {}
        self.opening_stats = {}  # To store win/loss statistics for each opening

    def add_opening(self, moves, variant="standard"):
        if variant == "standard":
            board = chess.Board()
        elif variant == "chess960":
            board = chess.Board.from_chess960_pos(random.randint(0, 959))
        elif variant == "kingofthehill":
            board = chess.variant.KingOfTheHillBoard()
        elif variant == "3check":
            board = chess.variant.ThreeCheckBoard()
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        key = (board.fen(), variant)
        for move in moves:
            if key not in self.openings:
                self.openings[key] = []
            self.openings[key].append(move)
            board.push_san(move)
            key = (board.fen(), variant)

    def get_move(self, board, variant="standard"):
        key = (board.fen(), variant)
        if key in self.openings:
            return random.choice(self.openings[key])
        return None

    def load_from_pgn(self, pgn_file, variant="standard"):
        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                moves = [move.san() for move in game.mainline_moves()]
                self.add_opening(moves[:10], variant)  # Add first 10 moves of each game

    def load_from_eco(self, eco_file):
        with open(eco_file, 'r') as f:
            for line in f:
                parts = line.strip().split('"')
                if len(parts) >= 4:
                    moves = parts[3].split()
                    self.add_opening(moves)

    def update_opening_stats(self, moves, result, variant="standard"):
        board = chess.Board() if variant == "standard" else chess.variant.find_variant(variant)()
        for move in moves:
            key = (board.fen(), variant)
            if key not in self.opening_stats:
                self.opening_stats[key] = {"wins": 0, "losses": 0, "draws": 0}
            if result == "1-0":
                self.opening_stats[key]["wins"] += 1
            elif result == "0-1":
                self.opening_stats[key]["losses"] += 1
            else:
                self.opening_stats[key]["draws"] += 1
            board.push_san(move)

    def get_best_move(self, board, variant="standard"):
        key = (board.fen(), variant)
        if key in self.openings:
            moves = self.openings[key]
            if key in self.opening_stats:
                stats = self.opening_stats[key]
                total_games = stats["wins"] + stats["losses"] + stats["draws"]
                if total_games > 0:
                    move_scores = [(move, (stats["wins"] + 0.5 * stats["draws"]) / total_games) for move in moves]
                    return max(move_scores, key=lambda x: x[1])[0]
            return random.choice(moves)
        return None

def create_sample_opening_book():
    book = OpeningBook()
    
    # Add some common openings for standard chess
    book.add_opening(["e4", "e5", "Nf3", "Nc6", "Bb5"])  # Ruy Lopez
    book.add_opening(["e4", "e5", "Nf3", "Nc6", "Bc4"])  # Italian Game
    book.add_opening(["d4", "d5", "c4"])  # Queen's Gambit
    book.add_opening(["e4", "c5"])  # Sicilian Defense
    book.add_opening(["e4", "e6"])  # French Defense
    book.add_opening(["e4", "c6"])  # Caro-Kann Defense
    book.add_opening(["e4", "d6"])  # Pirc Defense
    book.add_opening(["d4", "Nf6", "c4", "g6"])  # King's Indian Defense
    book.add_opening(["d4", "Nf6", "c4", "e6"])  # Nimzo-Indian Defense
    
    # Add some openings for Chess960
    book.add_opening(["e4", "e5", "Nf3", "Nc6"], variant="chess960")
    book.add_opening(["d4", "d5", "c4"], variant="chess960")
    
    # Add some openings for King of the Hill
    book.add_opening(["e4", "e5", "Nf3", "Nc6"], variant="kingofthehill")
    book.add_opening(["d4", "d5", "c4"], variant="kingofthehill")
    
    # Add some openings for Three-check chess
    book.add_opening(["e4", "e5", "Nf3", "Nc6"], variant="3check")
    book.add_opening(["d4", "d5", "c4"], variant="3check")
    
    return book

# Usage example
if __name__ == "__main__":
    book = create_sample_opening_book()
    
    # Test the opening book for standard chess
    board = chess.Board()
    for _ in range(3):
        move = book.get_best_move(board)
        if move:
            print(f"Suggested move: {move}")
            board.push_san(move)
        else:
            print("No move found in opening book")
            break
    
    print("Final position (standard chess):")
    print(board)
    
    # Test the opening book for Chess960
    board = chess.Board.from_chess960_pos(random.randint(0, 959))
    for _ in range(3):
        move = book.get_best_move(board, variant="chess960")
        if move:
            print(f"Suggested move (Chess960): {move}")
            board.push_san(move)
        else:
            print("No move found in opening book for Chess960")
            break
    
    print("Final position (Chess960):")
    print(board)
