import random

class OpeningBook:
    def __init__(self):
        self.openings = {
            'e2e4': ['e7e5', 'c7c5', 'e7e6', 'c7c6'],  # King's Pawn
            'e2e4 e7e5': ['g1f3', 'b1c3'],  # King's Pawn Game
            'e2e4 c7c5': ['g1f3', 'd2d4'],  # Sicilian Defense
            'e2e4 e7e6': ['d2d4', 'g1f3'],  # French Defense
            'd2d4': ['d7d5', 'g8f6', 'e7e6'],  # Queen's Pawn
            'd2d4 d7d5': ['c2c4', 'g1f3'],  # Queen's Gambit
            'd2d4 g8f6': ['c2c4', 'g1f3'],  # Indian Defense
        }

    def get_move(self, move_history):
        move_string = ' '.join(self.move_to_string(move) for move in move_history)
        if move_string in self.openings:
            return self.string_to_move(random.choice(self.openings[move_string]))
        return None

    @staticmethod
    def move_to_string(move):
        from_row, from_col, to_row, to_col = move
        return f"{chr(97+from_col)}{8-from_row}{chr(97+to_col)}{8-to_row}"

    @staticmethod
    def string_to_move(move_string):
        from_col = ord(move_string[0]) - 97
        from_row = 8 - int(move_string[1])
        to_col = ord(move_string[2]) - 97
        to_row = 8 - int(move_string[3])
        return (from_row, from_col, to_row, to_col)
