import random
import torch
from .game import ChessGame
from .opening_book import OpeningBook
from .mcts import MCTS
from .model import ChessModel

class ChessAI:
    def __init__(self, difficulty='medium'):
        self.difficulty = difficulty
        self.num_simulations = self._get_num_simulations()
        self.opening_book = OpeningBook()
        self.model = ChessModel()  # You need to implement this class
        self.mcts = MCTS(self.model, self.num_simulations)

    def _get_num_simulations(self):
        if self.difficulty == 'easy':
            return 100
        elif self.difficulty == 'medium':
            return 500
        elif self.difficulty == 'hard':
            return 1000
        else:
            return 500  # Default to medium

    def get_best_move(self, game):
        # First, check the opening book
        book_move = self.opening_book.get_move(game.move_history)
        if book_move:
            return book_move

        # If no book move, use AlphaZero (MCTS + Neural Network)
        state = game.get_state()
        action_probs = self.mcts.search(state)
        return self.select_move(game, action_probs)

    def select_move(self, game, action_probs):
        legal_moves = game.get_legal_moves()
        best_move = None
        best_prob = -float('inf')
        
        for move in legal_moves:
            move_prob = action_probs[move]
            if move_prob > best_prob:
                best_prob = move_prob
                best_move = move
        
        return best_move

    def get_random_move(self, game):
        legal_moves = game.get_legal_moves()
        return random.choice(legal_moves) if legal_moves else None

    def get_move(self, game):
        if self.difficulty == 'easy':
            # 70% chance of choosing a random move
            if random.random() < 0.7:
                return self.get_random_move(game)
        return self.get_best_move(game)
