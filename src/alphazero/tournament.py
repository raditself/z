
import torch
import os
import argparse
from tqdm import tqdm
from .game import Game
from .model import ChessModel
from .improved_model import ImprovedChessModel
from .mcts import MCTS
import numpy as np

def load_model(model_path, model_class=ImprovedChessModel):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    game = Game()
    model = model_class(board_size=game.board_size, action_size=game.action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def play_game(model1, model2, game, num_simulations=100):
    mcts1 = MCTS(game, model1, num_simulations)
    mcts2 = MCTS(game, model2, num_simulations)
    
    state = game.get_initial_state()
    player = 1
    
    while not game.is_game_over(state):
        if player == 1:
            action = mcts1.get_action(state)
        else:
            action = mcts2.get_action(state)
        
        state = game.get_next_state(state, action, player)
        player = -player
    
    return game.get_winner(state)

def run_tournament(model_paths, num_games=100):
    game = Game()
    models = [load_model(path) for path in model_paths]
    n = len(models)
    scores = np.zeros((n, n))
    
    total_games = n * (n - 1) * num_games // 2
    with tqdm(total=total_games, desc="Tournament Progress") as pbar:
        for i in range(n):
            for j in range(i+1, n):
                for _ in range(num_games // 2):
                    result = play_game(models[i], models[j], game)
                    if result == 1:
                        scores[i, j] += 1
                    elif result == -1:
                        scores[j, i] += 1
                    else:
                        scores[i, j] += 0.5
                        scores[j, i] += 0.5
                    
                    # Play reverse game
                    result = play_game(models[j], models[i], game)
                    if result == 1:
                        scores[j, i] += 1
                    elif result == -1:
                        scores[i, j] += 1
                    else:
                        scores[i, j] += 0.5
                        scores[j, i] += 0.5
                    
                    pbar.update(2)
    
    return scores

def calculate_elo(scores, k=32):
    n = len(scores)
    elo_ratings = [1500] * n
    
    for _ in range(10):  # Run multiple iterations for better convergence
        for i in range(n):
            for j in range(i+1, n):
                ea = 1 / (1 + 10 ** ((elo_ratings[j] - elo_ratings[i]) / 400))
                eb = 1 / (1 + 10 ** ((elo_ratings[i] - elo_ratings[j]) / 400))
                
                total_games = scores[i, j] + scores[j, i]
                if total_games > 0:
                    sa = scores[i, j] / total_games
                    sb = scores[j, i] / total_games
                    
                    elo_ratings[i] += k * (sa - ea)
                    elo_ratings[j] += k * (sb - eb)
    
    return elo_ratings

def main():
    parser = argparse.ArgumentParser(description="Run a tournament between different AlphaZero models")
    parser.add_argument("model_paths", nargs="+", help="Paths to the model files")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to play between each pair of models")
    args = parser.parse_args()

    scores = run_tournament(args.model_paths, args.num_games)
    elo_ratings = calculate_elo(scores)
    
    print("\nTournament Results:")
    for i, path in enumerate(args.model_paths):
        print(f"{path}: ELO Rating = {elo_ratings[i]:.2f}")

if __name__ == "__main__":
    main()
