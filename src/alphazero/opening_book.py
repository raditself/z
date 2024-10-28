
import random
import json
from src.games.chess import ChessGame
from src.games.checkers import CheckersGame

class OpeningBook:
    def __init__(self, game_type='chess'):
        self.game_type = game_type
        self.book = {}
        if game_type == 'chess':
            self.game = ChessGame()
        elif game_type == 'checkers':
            self.game = CheckersGame()
        else:
            raise ValueError("Invalid game type. Choose 'chess' or 'checkers'.")

    def add_opening(self, moves):
        current_position = self.game.get_initial_board()
        for move in moves:
            position_key = self.get_position_key(current_position)
            if position_key not in self.book:
                self.book[position_key] = []
            if move not in self.book[position_key]:
                self.book[position_key].append(move)
            current_position = self.game.make_move(current_position, move)

    def get_move(self, position):
        position_key = self.get_position_key(position)
        if position_key in self.book:
            return random.choice(self.book[position_key])
        return None

    def get_position_key(self, position):
        if self.game_type == 'chess':
            return position.fen()
        elif self.game_type == 'checkers':
            return ''.join(map(str, position.flatten()))

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.book, f)

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            self.book = json.load(f)

    def merge_books(self, other_book):
        for position_key, moves in other_book.book.items():
            if position_key not in self.book:
                self.book[position_key] = moves
            else:
                self.book[position_key] = list(set(self.book[position_key] + moves))

# Usage example for Chess:
# chess_book = OpeningBook('chess')
# chess_book.add_opening(["e2e4", "e7e5", "g1f3", "b8c6"])
# chess_book.add_opening(["d2d4", "d7d5", "c2c4", "e7e6"])
# chess_book.save_to_file("chess_openings.json")

# Usage example for Checkers:
# checkers_book = OpeningBook('checkers')
# checkers_book.add_opening([(2, 1, 3, 2), (5, 2, 4, 3), (1, 2, 2, 3)])
# checkers_book.add_opening([(2, 3, 3, 2), (5, 2, 4, 3), (3, 2, 4, 1)])
# checkers_book.save_to_file("checkers_openings.json")

# To use the opening book in the AI:
# if game.get_move_count(board) < 10:  # Only use opening book for first 10 moves
#     book_move = opening_book.get_move(board)
#     if book_move:
#         return book_move
# ... (continue with regular AI move selection)
