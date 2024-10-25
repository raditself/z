import random
import json
from tqdm import tqdm
from .game import ChessGame
from .model import AlphaZeroNetwork, import_model
from .mcts import MCTS

class Tournament:
    def __init__(self, model_paths, num_games=100, time_limit=600, increment=10):
        self.model_paths = model_paths
        self.num_games = num_games
        self.time_limit = time_limit
        self.increment = increment
        self.results = {path: {'wins': 0, 'losses': 0, 'draws': 0} for path in model_paths}

    def run_tournament(self):
        for i in tqdm(range(self.num_games), desc="Tournament Progress"):
            white_model_path, black_model_path = random.sample(self.model_paths, 2)
            result = self.play_game(white_model_path, black_model_path)
            self.update_results(white_model_path, black_model_path, result)

    def play_game(self, white_model_path, black_model_path):
        game = ChessGame(initial_time=self.time_limit, increment=self.increment)
        try:
            white_model = self.load_model(white_model_path)
            black_model = self.load_model(black_model_path)
        except Exception as e:
            print(f"Error loading models: {e}")
            return None

        white_mcts = MCTS(white_model)
        black_mcts = MCTS(black_model)

        while not game.is_game_over():
            if game.current_player == 1:
                move = white_mcts.get_best_move(game)
            else:
                move = black_mcts.get_best_move(game)
            game.make_move(move)

        return game.get_result()

    def load_model(self, model_path):
        try:
            input_shape = (3, 8, 8)  # Adjust if necessary
            model = AlphaZeroNetwork(input_shape, 64 * 64)  # 64*64 for all possible moves
            import_model(model, model_path)
            return model
        except Exception as e:
            raise Exception(f"Failed to load model from {model_path}: {e}")

    def update_results(self, white_model_path, black_model_path, result):
        if result is None:
            return
        if result == '1-0':
            self.results[white_model_path]['wins'] += 1
            self.results[black_model_path]['losses'] += 1
        elif result == '0-1':
            self.results[white_model_path]['losses'] += 1
            self.results[black_model_path]['wins'] += 1
        else:  # Draw
            self.results[white_model_path]['draws'] += 1
            self.results[black_model_path]['draws'] += 1

    def get_rankings(self):
        def score(results):
            return results['wins'] + 0.5 * results['draws']

        return sorted(self.results.items(), key=lambda x: score(x[1]), reverse=True)

    def print_results(self):
        rankings = self.get_rankings()
        print("Tournament Results:")
        for rank, (model_path, results) in enumerate(rankings, 1):
            print(f"{rank}. {model_path}: Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}")

    def save_results(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)

    def load_results(self, filename):
        with open(filename, 'r') as f:
            self.results = json.load(f)
