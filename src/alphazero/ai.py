
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
        self.model = ChessModel()
        self.mcts = MCTS(self.model, self.num_simulations)
        self.move_evaluations = {}  # New attribute to store move evaluations

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
            self.move_evaluations = {'book_move': book_move}
            return book_move

        # If no book move, use AlphaZero (MCTS + Neural Network)
        state = game.get_state()
        action_probs = self.mcts.search(state)
        
        # Store MCTS evaluations
        self.move_evaluations = {move: prob for move, prob in action_probs.items()}
        
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
        
        # Add best move evaluation
        self.move_evaluations['best_move'] = (best_move, best_prob)
        
        return best_move

    def get_random_move(self, game):
        legal_moves = game.get_legal_moves()
        random_move = random.choice(legal_moves) if legal_moves else None
        self.move_evaluations = {'random_move': random_move}
        return random_move

    def get_move(self, game):
        if self.difficulty == 'easy':
            # 70% chance of choosing a random move
            if random.random() < 0.7:
                return self.get_random_move(game)
        return self.get_best_move(game)

    def get_move_explanations(self):
        """
        Returns a dictionary containing explanations for the AI's move selection process.
        """
        explanations = {}
        
        if 'book_move' in self.move_evaluations:
            explanations['type'] = 'Opening Book'
            explanations['move'] = self.move_evaluations['book_move']
            explanations['explanation'] = "This move was selected from the opening book."
        elif 'random_move' in self.move_evaluations:
            explanations['type'] = 'Random Move'
            explanations['move'] = self.move_evaluations['random_move']
            explanations['explanation'] = "This move was selected randomly due to the 'easy' difficulty setting."
        else:
            explanations['type'] = 'MCTS Evaluation'
            best_move, best_prob = self.move_evaluations['best_move']
            explanations['move'] = best_move
            explanations['probability'] = best_prob
            explanations['top_alternatives'] = sorted(
                [(move, prob) for move, prob in self.move_evaluations.items() if move != 'best_move'],
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5 alternative moves
            explanations['explanation'] = f"This move was selected as the best move after MCTS evaluation. It had a probability of {best_prob:.2f}."
        
        return explanations

