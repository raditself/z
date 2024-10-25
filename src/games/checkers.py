
import numpy as np

class Checkers:
    def __init__(self):
        self.board_size = 8
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.initialize_board()
        self.current_player = 1
        self.move_directions = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)], dtype=np.int8)
        self.move_cache = {}

    def initialize_board(self):
        self.board[::2, 1::2] = 2
        self.board[1::2, ::2] = 2
        self.board[5::2, ::2] = 1
        self.board[6::2, 1::2] = 1

    def get_valid_moves(self):
        if self.current_player in self.move_cache:
            return self.move_cache[self.current_player]

        moves = []
        pieces = np.argwhere(self.board == self.current_player)
        for piece in pieces:
            moves.extend(self.get_piece_moves(*piece))

        self.move_cache[self.current_player] = moves
        return moves

    def get_piece_moves(self, row, col):
        moves = []
        new_positions = np.array([row, col]) + self.move_directions
        valid_positions = (new_positions >= 0) & (new_positions < self.board_size)
        valid_positions = valid_positions.all(axis=1)
        new_positions = new_positions[valid_positions]

        for new_row, new_col in new_positions:
            if self.board[new_row, new_col] == 0:
                moves.append((row, col, new_row, new_col))
            elif self.board[new_row, new_col] != self.current_player:
                jump_row, jump_col = new_row + (new_row - row), new_col + (new_col - col)
                if 0 <= jump_row < self.board_size and 0 <= jump_col < self.board_size and self.board[jump_row, jump_col] == 0:
                    moves.append((row, col, jump_row, jump_col))

        return moves

    def make_move(self, move):
        start_row, start_col, end_row, end_col = move
        self.board[end_row, end_col] = self.board[start_row, start_col]
        self.board[start_row, start_col] = 0
        if abs(start_row - end_row) == 2:  # Capture move
            captured_row, captured_col = (start_row + end_row) // 2, (start_col + end_col) // 2
            self.board[captured_row, captured_col] = 0
        self.current_player = 3 - self.current_player  # Switch player (1 -> 2, 2 -> 1)
        self.move_cache.clear()  # Clear the move cache after making a move

    def is_game_over(self):
        return len(self.get_valid_moves()) == 0

    def get_winner(self):
        if not self.is_game_over():
            return None
        return 3 - self.current_player  # The player who can't move loses

    def get_state(self):
        return self.board.copy()

    def get_current_player(self):
        return self.current_player

    def undo_move(self, move):
        start_row, start_col, end_row, end_col = move
        self.board[start_row, start_col] = self.board[end_row, end_col]
        self.board[end_row, end_col] = 0
        if abs(start_row - end_row) == 2:  # Capture move
            captured_row, captured_col = (start_row + end_row) // 2, (start_col + end_col) // 2
            self.board[captured_row, captured_col] = 3 - self.current_player  # Restore captured piece
        self.current_player = 3 - self.current_player  # Switch back to the previous player
        self.move_cache.clear()  # Clear the move cache after undoing a move

    def __str__(self):
        symbols = {0: '.', 1: 'x', 2: 'o'}
        return '\n'.join(' '.join(symbols[piece] for piece in row) for row in self.board)
