import numpy as np

class Checkers:
    def __init__(self):
        self.board_size = 8
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.initialize_board()
        self.current_player = 1
        self.move_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def initialize_board(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i + j) % 2 == 1:
                    if i < 3:
                        self.board[i, j] = 2  # Player 2 pieces
                    elif i > 4:
                        self.board[i, j] = 1  # Player 1 pieces

    def get_valid_moves(self):
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == self.current_player:
                    moves.extend(self.get_piece_moves(i, j))
        return moves

    def get_piece_moves(self, row, col):
        moves = []
        for dr, dc in self.move_directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col) and self.board[new_row, new_col] == 0:
                moves.append((row, col, new_row, new_col))
            elif self.is_valid_position(new_row, new_col) and self.board[new_row, new_col] != self.current_player:
                jump_row, jump_col = new_row + dr, new_col + dc
                if self.is_valid_position(jump_row, jump_col) and self.board[jump_row, jump_col] == 0:
                    moves.append((row, col, jump_row, jump_col))
        return moves

    def is_valid_position(self, row, col):
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def make_move(self, move):
        start_row, start_col, end_row, end_col = move
        self.board[end_row, end_col] = self.board[start_row, start_col]
        self.board[start_row, start_col] = 0
        if abs(start_row - end_row) == 2:  # Capture move
            captured_row, captured_col = (start_row + end_row) // 2, (start_col + end_col) // 2
            self.board[captured_row, captured_col] = 0
        self.current_player = 3 - self.current_player  # Switch player (1 -> 2, 2 -> 1)

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

    def __str__(self):
        symbols = {0: '.', 1: 'x', 2: 'o'}
        return '\n'.join(' '.join(symbols[piece] for piece in row) for row in self.board)
