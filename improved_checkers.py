import numpy as np
import random

class ImprovedCheckersBoard:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.kings = np.zeros((8, 8), dtype=int)
        self.move_log = []
        self.setup_board()

    def setup_board(self):
        for i in range(3):
            for j in range(8):
                if (i + j) % 2 == 1:
                    self.board[i][j] = 1
        for i in range(5, 8):
            for j in range(8):
                if (i + j) % 2 == 1:
                    self.board[i][j] = -1

    def make_move(self, move):
        i, j, new_i, new_j = move
        self.kings[new_i][new_j] = self.kings[i][j]
        self.board[new_i][new_j] = self.board[i][j]
        self.board[i][j] = 0
        self.kings[i][j] = 0
        
        # Handle king promotion
        promotion = False
        if (new_i == 0 and self.board[new_i][new_j] == 1) or (new_i == 7 and self.board[new_i][new_j] == -1):
            self.kings[new_i][new_j] = 1
            promotion = True
        
        # Handle captures
        captured = []
        if abs(new_i - i) == 2:
            captured_i, captured_j = (i + new_i) // 2, (j + new_j) // 2
            captured.append((captured_i, captured_j))
            self.board[captured_i][captured_j] = 0
            self.kings[captured_i][captured_j] = 0
        
        # Log the move
        self.move_log.append({
            'move': move,
            'player': self.board[new_i][new_j],
            'is_king': bool(self.kings[new_i][new_j]),
            'captured': captured,
            'promotion': promotion
        })
        
        return self

    def get_legal_moves(self):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == 1 or (self.board[i][j] == -1 and self.kings[i][j]):
                    moves.extend(self._get_moves(i, j, 1))
                if self.board[i][j] == -1 or (self.board[i][j] == 1 and self.kings[i][j]):
                    moves.extend(self._get_moves(i, j, -1))
        return moves

    def _get_moves(self, i, j, player):
        moves = []
        directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)] if self.kings[i][j] else [(player, -1), (player, 1)]
        for di, dj in directions:
            if 0 <= i + di < 8 and 0 <= j + dj < 8:
                if self.board[i + di][j + dj] == 0:
                    moves.append((i, j, i + di, j + dj))
                elif 0 <= i + 2*di < 8 and 0 <= j + 2*dj < 8 and self.board[i + 2*di][j + 2*dj] == 0:
                    if self.board[i + di][j + dj] == -player:
                        moves.append((i, j, i + 2*di, j + 2*dj))
        return moves

    def is_game_over(self):
        return len(self.get_legal_moves()) == 0

    def get_winner(self):
        if np.sum(self.board == 1) == 0:
            return -1
        elif np.sum(self.board == -1) == 0:
            return 1
        elif len(self.get_legal_moves()) == 0:
            return -1 if np.sum(self.board == 1) < np.sum(self.board == -1) else 1
        else:
            return 0

    def evaluate(self):
        player_1_pieces = np.sum(self.board == 1)
        player_2_pieces = np.sum(self.board == -1)
        player_1_kings = np.sum(self.kings[self.board == 1])
        player_2_kings = np.sum(self.kings[self.board == -1])
        
        player_1_back_row = np.sum(self.board[7] == 1)
        player_2_back_row = np.sum(self.board[0] == -1)
        
        player_1_center = np.sum(self.board[3:5, 2:6] == 1)
        player_2_center = np.sum(self.board[3:5, 2:6] == -1)
        
        score = (player_1_pieces - player_2_pieces) * 10 +                 (player_1_kings - player_2_kings) * 30 +                 (player_1_center - player_2_center) * 5 +                 (player_1_back_row - player_2_back_row) * 5
        
        return score

def alpha_beta_search(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return board.evaluate()
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in board.get_legal_moves():
            new_board = ImprovedCheckersBoard()
            new_board.board = board.board.copy()
            new_board.kings = board.kings.copy()
            new_board.make_move(move)
            eval = alpha_beta_search(new_board, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.get_legal_moves():
            new_board = ImprovedCheckersBoard()
            new_board.board = board.board.copy()
            new_board.kings = board.kings.copy()
            new_board.make_move(move)
            eval = alpha_beta_search(new_board, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board, player):
    depth = 4
    best_move = None
    best_value = float('-inf') if player == 1 else float('inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for move in board.get_legal_moves():
        new_board = ImprovedCheckersBoard()
        new_board.board = board.board.copy()
        new_board.kings = board.kings.copy()
        new_board.make_move(move)
        value = alpha_beta_search(new_board, depth - 1, alpha, beta, player != 1)
        
        if player == 1 and value > best_value:
            best_value = value
            best_move = move
            alpha = max(alpha, value)
        elif player == -1 and value < best_value:
            best_value = value
            best_move = move
            beta = min(beta, value)
    
    return best_move

def play_game():
    board = ImprovedCheckersBoard()
    player = 1
    moves = 0
    
    while not board.is_game_over() and moves < 200:
        move = get_best_move(board, player)
        if move is None:
            break
        
        board.make_move(move)
        player = -player
        moves += 1
    
    winner = board.get_winner()
    return winner, moves, board.move_log

# Example usage
winner, total_moves, move_log = play_game()
print(f"Game result: {'Player 1' if winner == 1 else 'Player -1' if winner == -1 else 'Draw'}")
print(f"Total moves: {total_moves}")

king_promotions = sum(1 for move in move_log if move['promotion'])
captures = sum(len(move['captured']) for move in move_log)

print(f"\nGame Statistics:")
print(f"King Promotions: {king_promotions}")
print(f"Total Captures: {captures}")

