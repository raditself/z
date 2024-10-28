
import time
import numpy as np
from src.games.chess import Chess
from src.games.go import Go
from src.games.shogi import Shogi
from src.games.othello import Othello
from src.games.connect_four import ConnectFour
from src.alphazero.mcts import AdaptiveMCTS
from src.alphazero.neural_network import DynamicNeuralNetwork

class Stockfish:
    def __init__(self, elo=2000):
        self.elo = elo
    
    def get_move(self, state):
        # Simulate Stockfish move selection
        return np.random.choice(Chess().get_valid_actions(state))

class LeelaZero:
    def get_move(self, state):
        # Simulate Leela Zero move selection
        return np.random.choice(Go().get_valid_actions(state))

class ElmoShogi:
    def get_move(self, state):
        # Simulate Elmo move selection
        return np.random.choice(Shogi().get_valid_actions(state))

class Edax:
    def get_move(self, state):
        # Simulate Edax move selection
        return np.random.choice(Othello().get_valid_actions(state))

class Solver:
    def get_move(self, state):
        # Simulate perfect Connect Four solver
        return np.random.choice(ConnectFour().get_valid_actions(state))

def play_game(game, player1, player2, max_moves=1000):
    state = game.get_initial_state()
    for _ in range(max_moves):
        if game.is_terminal(state):
            return game.get_reward(state)
        
        if game.current_player == 1:
            action = player1.get_move(state)
        else:
            action = player2.get_move(state)
        
        state = game.get_next_state(state, action)
        game.current_player *= -1
    
    return 0  # Draw if max moves reached

def benchmark(game, alphazero, opponent, num_games=100):
    start_time = time.time()
    results = []
    
    for _ in range(num_games):
        if np.random.choice([True, False]):
            result = play_game(game, alphazero, opponent)
        else:
            result = -play_game(game, opponent, alphazero)
        results.append(result)
    
    win_rate = (np.array(results) > 0).mean()
    draw_rate = (np.array(results) == 0).mean()
    loss_rate = (np.array(results) < 0).mean()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate,
        'total_time': total_time,
        'avg_time_per_game': total_time / num_games
    }

def run_benchmarks():
    games = [Chess(), Go(), Shogi(), Othello(), ConnectFour()]
    opponents = [Stockfish(), LeelaZero(), ElmoShogi(), Edax(), Solver()]
    
    for game, opponent in zip(games, opponents):
        print(f"Benchmarking {game.__class__.__name__} against {opponent.__class__.__name__}")
        
        network = DynamicNeuralNetwork(game)
        alphazero = AdaptiveMCTS(game, network)
        
        results = benchmark(game, alphazero, opponent)
        
        print(f"Results:")
        print(f"Win rate: {results['win_rate']:.2f}")
        print(f"Draw rate: {results['draw_rate']:.2f}")
        print(f"Loss rate: {results['loss_rate']:.2f}")
        print(f"Total time: {results['total_time']:.2f} seconds")
        print(f"Average time per game: {results['avg_time_per_game']:.2f} seconds")
        print()

if __name__ == "__main__":
    run_benchmarks()
