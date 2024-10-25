class CheckersGame:
    def __init__(self):
        self.board = self.get_initial_board()
        self.current_player = 1  # 1 for white, -1 for black

    def get_initial_board(self):
        board = [[0] * 8 for _ in range(8)]
        for i in range(3):
            for j in range(8):
                if (i + j) % 2 == 1:
                    board[i][j] = 1  # white pieces
        for i in range(5, 8):
            for j in range(8):
                if (i + j) % 2 == 1:
                    board[i][j] = -1  # black pieces
        return board

    def get_valid_moves(self, player):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == player:
                    moves.extend(self.get_piece_moves(i, j))
        return moves

    def get_piece_moves(self, row, col):
        moves = []
        directions = [(1, -1), (1, 1)] if self.board[row][col] == 1 else [(-1, -1), (-1, 1)]
        for d_row, d_col in directions:
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < 8 and 0 <= new_col < 8 and self.board[new_row][new_col] == 0:
                moves.append((row, col, new_row, new_col))
        return moves

    def make_move(self, move):
        start_row, start_col, end_row, end_col = move
        self.board[end_row][end_col] = self.board[start_row][start_col]
        self.board[start_row][start_col] = 0
        self.current_player *= -1

    def is_game_over(self):
        return len(self.get_valid_moves(1)) == 0 or len(self.get_valid_moves(-1)) == 0

    def get_winner(self):
        if len(self.get_valid_moves(1)) == 0:
            return -1
        elif len(self.get_valid_moves(-1)) == 0:
            return 1
        else:
            return 0  # game is not over

