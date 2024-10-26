
import numpy as np

class GoGame:
    def __init__(self, board_size=19):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.ko = None
        self.game_over = False
        self.passes = 0

    def play(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row, col] = self.current_player
            self.remove_captured_stones(row, col)
            self.check_ko(row, col)
            self.current_player = 3 - self.current_player
            self.passes = 0
            return True
        return False

    def pass_turn(self):
        self.passes += 1
        if self.passes == 2:
            self.game_over = True
        self.current_player = 3 - self.current_player

    def is_valid_move(self, row, col):
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        if self.board[row, col] != 0:
            return False
        if (row, col) == self.ko:
            return False
        # Check for suicide rule
        test_board = self.board.copy()
        test_board[row, col] = self.current_player
        if self.has_liberties(test_board, row, col):
            return True
        return False

    def has_liberties(self, board, row, col):
        color = board[row, col]
        visited = set()
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if board[nr, nc] == 0:
                        return True
                    if board[nr, nc] == color:
                        stack.append((nr, nc))
        return False

    def remove_captured_stones(self, row, col):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.board[nr, nc] == 3 - self.current_player:
                    if not self.has_liberties(self.board, nr, nc):
                        self.remove_group(nr, nc)

    def remove_group(self, row, col):
        color = self.board[row, col]
        stack = [(row, col)]
        removed = []
        while stack:
            r, c = stack.pop()
            if self.board[r, c] != color:
                continue
            self.board[r, c] = 0
            removed.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self.board[nr, nc] == color:
                        stack.append((nr, nc))
        return removed

    def check_ko(self, row, col):
        captures = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.board[nr, nc] == 3 - self.current_player:
                    if not self.has_liberties(self.board, nr, nc):
                        captures.extend(self.remove_group(nr, nc))
        if len(captures) == 1 and not self.has_liberties(self.board, row, col):
            self.ko = captures[0]
        else:
            self.ko = None

    def get_legal_moves(self):
        moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_valid_move(row, col):
                    moves.append((row, col))
        return moves

    def get_winner(self):
        if not self.game_over:
            return None
        black_score = np.sum(self.board == 1)
        white_score = np.sum(self.board == 2)
        if black_score > white_score:
            return 1
        elif white_score > black_score:
            return 2
        else:
            return 0  # Draw

    def __str__(self):
        board_str = ""
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == 0:
                    board_str += ". "
                elif self.board[row, col] == 1:
                    board_str += "B "
                else:
                    board_str += "W "
            board_str += "\n"
        return board_str

# Test the Go game implementation
if __name__ == "__main__":
    game = GoGame(board_size=9)
    print(game)
    
    # Make some moves
    game.play(2, 2)
    game.play(3, 3)
    game.play(2, 3)
    game.play(3, 2)
    
    print(game)
    
    print("Legal moves:", game.get_legal_moves())
    print("Current player:", game.current_player)
    print("Game over:", game.game_over)
