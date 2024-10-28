
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from src.games.chess import Chess
from src.games.go import Go
from src.games.shogi import Shogi
from src.games.othello import Othello
from src.games.connect_four import ConnectFour
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

def train_network(config, checkpoint_dir=None):
    game = config["game"]
    network = DynamicNeuralNetwork(game)
    optimizer = torch.optim.Adam(network.parameters(), lr=config["lr"])

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        network.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for iteration in range(config["num_iterations"]):
        game_histories = ray.get([SelfPlayWorker.remote(game, network).play_game.remote() 
                                  for _ in range(config["games_per_iteration"])])
        
        for batch in process_game_data(game_histories):
            loss = network.train_on_batch(batch)
        
        if iteration % 10 == 0:
            with tune.checkpoint_dir(step=iteration) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((network.state_dict(), optimizer.state_dict()), path)
            
            tune.report(loss=loss)

def process_game_data(game_histories):
    # Process and yield batches of training data
    pass

def optimize_hyperparameters():
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        perturbation_interval=5,
        hyperparam_mutations={
            "lr": lambda: np.random.uniform(1e-4, 1e-2),
            "num_mcts_sims": lambda: np.random.randint(100, 1000),
            "c_puct": lambda: np.random.uniform(1, 5),
        })

    games = [Chess(), Go(), Shogi(), Othello(), ConnectFour()]
    
    for game in games:
        analysis = tune.run(
            train_network,
            name=f"alphazero_{game.__class__.__name__}",
            scheduler=scheduler,
            num_samples=4,
            config={
                "game": game,
                "lr": 1e-3,
                "num_mcts_sims": 800,
                "c_puct": 1.0,
                "num_iterations": 100,
                "games_per_iteration": 100,
            },
            resources_per_trial={"cpu": 8, "gpu": 1},
            stop={"training_iteration": 100},
        )

        print(f"Best hyperparameters for {game.__class__.__name__}:", analysis.best_config)

if __name__ == "__main__":
    ray.init()
    optimize_hyperparameters()
    ray.shutdown()
