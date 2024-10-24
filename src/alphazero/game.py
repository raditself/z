
import numpy as np

class ChessGame:
    def __init__(self):
        self.board = self.init_board()
        self.current_player = 1  # 1 for white, -1 for black

    def init_board(self):
        # Initialize the chess board
        board = np.zeros((8, 8, 6), dtype=np.int8)
        # Set up initial positions (simplified for this example)
        # 0: Pawn, 1: Rook, 2: Knight, 3: Bishop, 4: Queen, 5: King
        board[1] = board[6] = [1, 0, 0, 0, 0, 0]  # Pawns
        board[0] = board[7] = [0, 1, 1, 1, 1, 1]  # Other pieces
        board[7] *= -1
        board[6] *= -1
        return board

    def get_legal_moves(self):
        # Simplified legal moves (only considers pawns moving forward)
        legal_moves = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j][0] == self.current_player:
                    if self.current_player == 1 and i < 7 and self.board[i+1][j].sum() == 0:
                        legal_moves.append((i, j, i+1, j))
                    elif self.current_player == -1 and i > 0 and self.board[i-1][j].sum() == 0:
                        legal_moves.append((i, j, i-1, j))
        return legal_moves

    def make_move(self, move):
        from_row, from_col, to_row, to_col = move
        self.board[to_row][to_col] = self.board[from_row][from_col]
        self.board[from_row][from_col] = 0
        self.current_player *= -1

    def is_game_over(self):
        # Simplified game over condition (no kings left)
        return 5 not in self.board[:, :, 5] or -5 not in self.board[:, :, 5]

    def get_winner(self):
        if 5 not in self.board[:, :, 5]:
            return -1  # Black wins
        elif -5 not in self.board[:, :, 5]:
            return 1  # White wins
        return 0  # Draw or game not over

    def get_state(self):
        return self.board.copy(), self.current_player
