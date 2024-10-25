
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

    def evaluate_position(self):
        if self.is_game_over():
            winner = self.get_winner()
            if winner == 1:
                return 1000
            elif winner == 2:
                return -1000
            else:
                return 0

        piece_count = np.sum(self.board == 1) - np.sum(self.board == 2)
        king_count = np.sum(self.board == 3) - np.sum(self.board == 4)  # Assuming 3 for player 1 kings, 4 for player 2 kings

        # Evaluate piece positioning
        player1_positions = np.argwhere(self.board == 1)
        player2_positions = np.argwhere(self.board == 2)
        player1_avg_rank = np.mean(player1_positions[:, 0]) if len(player1_positions) > 0 else 0
        player2_avg_rank = np.mean(player2_positions[:, 0]) if len(player2_positions) > 0 else 0
        position_score = player2_avg_rank - player1_avg_rank  # Player 1 wants to minimize this, Player 2 wants to maximize

        # Control of key squares (corners and center)
        key_squares = [(0, 0), (0, 7), (7, 0), (7, 7), (3, 3), (3, 4), (4, 3), (4, 4)]
        key_square_control = sum(self.board[r, c] == 1 for r, c in key_squares) - sum(self.board[r, c] == 2 for r, c in key_squares)

        # Mobility (number of legal moves)
        current_player_moves = len(self.get_valid_moves())
        self.current_player = 3 - self.current_player
        opponent_moves = len(self.get_valid_moves())
        self.current_player = 3 - self.current_player
        mobility = current_player_moves - opponent_moves

        # Combine all factors with appropriate weights
        evaluation = (
            3 * piece_count +
            5 * king_count +
            0.5 * position_score +
            2 * key_square_control +
            0.1 * mobility
        )

        return evaluation if self.current_player == 1 else -evaluation

    def get_dynamic_search_depth(self, base_depth=4):
        # Count the number of pieces on the board
        total_pieces = np.sum(self.board != 0)
        
        # Adjust depth based on the number of pieces
        if total_pieces > 16:
            depth = base_depth
        elif total_pieces > 12:
            depth = base_depth + 1
        elif total_pieces > 8:
            depth = base_depth + 2
        else:
            depth = base_depth + 3

        # Adjust depth based on the phase of the game
        moves_played = np.sum(self.board == 0) - 32  # Assuming 32 empty squares at the start
        if moves_played < 10:
            depth = min(depth, max(base_depth - 1, 3))  # Reduce depth in the opening, but not below 3
        
        # Adjust depth based on the current evaluation
        current_eval = abs(self.evaluate_position())
        if current_eval > 500:  # If one side has a significant advantage
            depth = min(depth, base_depth - 1)  # Reduce depth to speed up inevitable win/loss
        
        return depth

import time

class CheckersAI:
    def __init__(self, game, max_time=5):
        self.game = game
        self.max_time = max_time

    def get_best_move(self):
        start_time = time.time()
        best_move = None
        depth = 1

        while time.time() - start_time < self.max_time:
            move, _ = self.minimax(depth, float('-inf'), float('inf'), True, start_time)
            if move is not None:
                best_move = move
            depth += 1

        return best_move

    def minimax(self, depth, alpha, beta, maximizing_player, start_time):
        if depth == 0 or self.game.is_game_over() or time.time() - start_time > self.max_time:
            return None, self.game.evaluate_position()

        best_move = None
        if maximizing_player:
            max_eval = float('-inf')
            for move in self.game.get_valid_moves():
                self.game.make_move(move)
                _, eval = self.minimax(depth - 1, alpha, beta, False, start_time)
                self.game.undo_move(move)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return best_move, max_eval
        else:
            min_eval = float('inf')
            for move in self.game.get_valid_moves():
                self.game.make_move(move)
                _, eval = self.minimax(depth - 1, alpha, beta, True, start_time)
                self.game.undo_move(move)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return best_move, min_eval
