import numpy as np
import random

class ImprovedCheckersBoard(CheckersBoard):
    def __init__(self):
        super().__init__()
        self.kings = np.zeros((8, 8), dtype=int)

    def make_move(self, move):
        i, j, new_i, new_j = move
        self.kings[new_i][new_j] = self.kings[i][j]
        self.board[new_i][new_j] = self.board[i][j]
        self.board[i][j] = 0
        self.kings[i][j] = 0
        
        # Handle king promotion
        if (new_i == 0 and self.board[new_i][new_j] == 1) or (new_i == 7 and self.board[new_i][new_j] == -1):
            self.kings[new_i][new_j] = 1
        
        # Handle captures
        if abs(new_i - i) == 2:
            self.board[(i + new_i) // 2][(j + new_j) // 2] = 0
            self.kings[(i + new_i) // 2][(j + new_j) // 2] = 0
        
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

    def evaluate(self):
        player_1_pieces = np.sum(self.board == 1)
        player_2_pieces = np.sum(self.board == -1)
        player_1_kings = np.sum(self.kings[self.board == 1])
        player_2_kings = np.sum(self.kings[self.board == -1])
        
        player_1_center = np.sum(self.board[3:5, 2:6] == 1)
        player_2_center = np.sum(self.board[3:5, 2:6] == -1)
        
        score = (player_1_pieces - player_2_pieces) * 10 +                 (player_1_kings - player_2_kings) * 30 +                 (player_1_center - player_2_center) * 5
        
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
    depth = 3  # Reduced depth for faster execution
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
    game_history = [board.board.copy()]
    kings_history = [board.kings.copy()]
    
    while not board.is_game_over() and moves < 100:  # Reduced max moves to 100
        move = get_best_move(board, player)
        if move is None:
            break
        
        board.make_move(move)
        player = -player
        moves += 1
        game_history.append(board.board.copy())
        kings_history.append(board.kings.copy())
    
    winner = board.get_winner()
    return winner, moves, game_history, kings_history

def print_board(board, kings):
    for i, row in enumerate(board):
        print(' '.join(['K' if kings[i][j] else '.XO'[int(piece) + 1] for j, piece in enumerate(row)]))
    print()

# Example usage
winner, total_moves, game_history, kings_history = play_game()

print("Initial board state:")
print_board(game_history[0], kings_history[0])

print("Middle game state:")
mid_game = len(game_history) // 2
print_board(game_history[mid_game], kings_history[mid_game])

print("Final board state:")
print_board(game_history[-1], kings_history[-1])

print(f"Game result: {'Player 1' if winner == 1 else 'Player -1' if winner == -1 else 'Draw'}")
print(f"Total moves: {total_moves}")
