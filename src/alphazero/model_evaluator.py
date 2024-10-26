
import os
from typing import List, Tuple
from .rating_system import RatingSystem, Player
from .alphazero import AlphaZero
from .game import Game

class ModelEvaluator:
    def __init__(self, game: Game, num_games: int = 100):
        self.game = game
        self.num_games = num_games
        self.rating_system = RatingSystem()
        self.models = {}

    def add_model(self, model_id: str, model: AlphaZero):
        self.models[model_id] = model
        self.rating_system.add_player(model_id)

    def evaluate_model(self, model_id: str, opponent_id: str) -> float:
        model = self.models[model_id]
        opponent = self.models[opponent_id]
        
        wins = 0
        for _ in range(self.num_games):
            result = self._play_game(model, opponent)
            if result == 1:
                wins += 1
            elif result == 0.5:
                wins += 0.5
        
        score = wins / self.num_games
        self.rating_system.update_ratings(model_id, opponent_id, score)
        return score

    def _play_game(self, model1: AlphaZero, model2: AlphaZero) -> float:
        # Implement game playing logic here
        # Return 1 for model1 win, 0 for model2 win, 0.5 for draw
        pass

    def get_model_rating(self, model_id: str) -> float:
        return self.rating_system.get_rating(model_id)

    def get_top_models(self, n: int = 5) -> List[Tuple[str, float]]:
        sorted_models = sorted(self.models.keys(), key=lambda x: self.get_model_rating(x), reverse=True)
        return [(model_id, self.get_model_rating(model_id)) for model_id in sorted_models[:n]]

    def save_ratings(self, filename: str):
        with open(filename, 'w') as f:
            for model_id in self.models:
                rating = self.get_model_rating(model_id)
                f.write(f"{model_id},{rating}\n")

    def load_ratings(self, filename: str):
        with open(filename, 'r') as f:
            for line in f:
                model_id, rating = line.strip().split(',')
                self.rating_system.add_player(model_id, float(rating))

# Example usage
if __name__ == "__main__":
    # This is just a placeholder. In a real scenario, you would use your actual Game and AlphaZero implementations.
    class DummyGame(Game):
        pass

    class DummyAlphaZero(AlphaZero):
        pass

    game = DummyGame()
    evaluator = ModelEvaluator(game)

    # Add some dummy models
    for i in range(5):
        model = DummyAlphaZero()
        evaluator.add_model(f"Model_{i}", model)

    # Evaluate models against each other
    for i in range(5):
        for j in range(i+1, 5):
            evaluator.evaluate_model(f"Model_{i}", f"Model_{j}")

    # Print top models
    print("Top models:")
    for model_id, rating in evaluator.get_top_models():
        print(f"{model_id}: {rating}")

    # Save and load ratings
    evaluator.save_ratings("model_ratings.csv")
    new_evaluator = ModelEvaluator(game)
    new_evaluator.load_ratings("model_ratings.csv")
