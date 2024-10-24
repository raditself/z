import random
from .game import ChessGame
from .opening_book import OpeningBook

class ChessAI:
    def __init__(self, difficulty='medium'):
        self.difficulty = difficulty
        self.max_depth = self._get_max_depth()
        self.move_time_limit = self._get_move_time_limit()
        self.opening_book = OpeningBook()

    def _get_max_depth(self):
        if self.difficulty == 'easy':
            return 2
        elif self.difficulty == 'medium':
            return 4
        elif self.difficulty == 'hard':
            return 6
        else:
            return 4  # Default to medium

    def _get_move_time_limit(self):
        if self.difficulty == 'easy':
            return 1  # 1 second
        elif self.difficulty == 'medium':
            return 3  # 3 seconds
        elif self.difficulty == 'hard':
            return 5  # 5 seconds
        else:
            return 3  # Default to medium

    def get_best_move(self, game):
        # First, check the opening book
        book_move = self.opening_book.get_move(game.move_history)
        if book_move:
            return book_move

        # If no book move, use minimax
        _, best_move = self.minimax(game, self.max_depth, float('-inf'), float('inf'), True)
        return best_move

    def minimax(self, game, depth, alpha, beta, maximizing_player):
        if depth == 0 or game.is_game_over():
            return game.evaluate(), None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in game.get_legal_moves():
                new_game = game.clone()
                new_game.make_move(move)
                eval, _ = self.minimax(new_game, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in game.get_legal_moves():
                new_game = game.clone()
                new_game.make_move(move)
                eval, _ = self.minimax(new_game, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def get_random_move(self, game):
        legal_moves = game.get_legal_moves()
        return random.choice(legal_moves) if legal_moves else None

    def get_move(self, game):
        if self.difficulty == 'easy':
            # 70% chance of choosing a random move
            if random.random() < 0.7:
                return self.get_random_move(game)
        return self.get_best_move(game)
