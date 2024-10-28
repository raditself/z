
import numpy as np

class CheckersOpeningBook:
    def __init__(self):
        self.openings = {
            # Format: 'board_state': (move, evaluation)
            # Board state is represented as a string of the first few moves
            '11-15': ((18, 15), 0.1),  # Single Corner Opening
            '11-15 23-19': ((8, 11), 0.2),
            '11-15 23-19 8-11': ((22, 18), 0.1),
            '9-13': ((21, 17), 0.0),  # Double Corner Opening
            '9-13 21-17': ((5, 9), 0.1),
            '9-13 21-17 5-9': ((23, 18), 0.0),
            '10-14': ((22, 18), 0.0),  # Crescent Opening
            '10-14 22-18': ((6, 10), 0.1),
            '10-14 22-18 6-10': ((24, 19), 0.0),
            '12-16': ((24, 20), 0.0),  # Pioneer Opening
            '12-16 24-20': ((8, 12), 0.1),
            '12-16 24-20 8-12': ((22, 18), 0.0),
        }

    def get_opening_move(self, board):
        board_state = self._board_to_state(board)
        if board_state in self.openings:
            return self.openings[board_state][0]
        return None

    def get_opening_evaluation(self, board):
        board_state = self._board_to_state(board)
        if board_state in self.openings:
            return self.openings[board_state][1]
        return None

    def _board_to_state(self, board):
        # Convert the board to a string representation of moves
        moves = []
        for i in range(32):
            row, col = i // 4, (i % 4) * 2 + (1 - i // 4 % 2)
            if board[row][col] != 0:
                moves.append(f"{i+1}-{board[row][col]}")
        return ' '.join(moves)

    def add_opening(self, board_state, move, evaluation):
        self.openings[board_state] = (move, evaluation)

    def remove_opening(self, board_state):
        if board_state in self.openings:
            del self.openings[board_state]
