
import ray
from src.games.chess import Chess
from src.alphazero.neural_network import DynamicNeuralNetwork
from src.alphazero.mcts import AdaptiveMCTS

@ray.remote
class SelfPlayWorker:
    def __init__(self, game, network):
        self.game = game
        self.network = network
        self.mcts = AdaptiveMCTS(self.game, self.network)

    def play_game(self):
        state = self.game.get_initial_state()
        game_history = []

        while not self.game.is_terminal(state):
            action_probs = self.mcts.get_action_prob(state)
            game_history.append((state, action_probs, self.game.current_player))
            action = self.mcts.search(state)
            state = self.game.get_next_state(state, action)

        return game_history

def distributed_self_play(num_workers, num_games_per_worker):
    ray.init()
    game = Chess()
    network = DynamicNeuralNetwork(game)

    workers = [SelfPlayWorker.remote(game, network) for _ in range(num_workers)]
    game_histories = []

    for _ in range(num_games_per_worker):
        game_futures = [worker.play_game.remote() for worker in workers]
        game_histories.extend(ray.get(game_futures))

    ray.shutdown()
    return game_histories

def train_network_distributed(network, game_histories):
    # Implement distributed network training here
    pass

if __name__ == "__main__":
    num_workers = 4
    num_games_per_worker = 10
    game_histories = distributed_self_play(num_workers, num_games_per_worker)
    
    game = Chess()
    network = DynamicNeuralNetwork(game)
    train_network_distributed(network, game_histories)

    print(f"Completed {num_workers * num_games_per_worker} games of distributed self-play and training.")
