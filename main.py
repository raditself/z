
import torch
from src.games.chess import Chess
from src.games.go import Go
from src.games.shogi import Shogi
from src.games.othello import Othello
from src.games.connect_four import ConnectFour
from src.alphazero.neural_network import DynamicNeuralNetwork
from src.alphazero.mcts import AdaptiveMCTS
from src.advanced_heuristics import AdvancedHeuristics
from src.optimized_distributed_training import optimize_hyperparameters
from src.benchmark_system import run_benchmarks
from src.logging_visualization import AlphaZeroLogger, plot_game_length_distribution, plot_action_heatmap
import numpy as np
import ray

def main():
    # Initialize logger
    logger = AlphaZeroLogger('logs')

    # Initialize games
    games = [Chess(), Go(), Shogi(), Othello(), ConnectFour()]

    # Initialize Ray for distributed computing
    ray.init()

    for game in games:
        print(f"Training AlphaZero for {game.__class__.__name__}")

        # Optimize hyperparameters
        best_config = optimize_hyperparameters(game)

        # Initialize neural network with optimized hyperparameters
        network = DynamicNeuralNetwork(game, **best_config)

        # Initialize MCTS with optimized hyperparameters
        mcts = AdaptiveMCTS(game, network, **best_config)

        # Training loop
        num_iterations = best_config['num_iterations']
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")

            # Distributed self-play
            game_histories = ray.get([mcts.self_play.remote() for _ in range(best_config['num_self_play_games'])])

            # Train network distributedly
            network.train_distributed(game_histories)

            # Apply advanced heuristics
            heuristic_value = getattr(AdvancedHeuristics, f"{game.__class__.__name__.lower()}_heuristics")(game.get_initial_state())
            print(f"Heuristic value: {heuristic_value}")

            # Log training progress
            loss, policy_loss, value_loss = network.get_loss_metrics()
            game_length = np.mean([len(history) for history in game_histories])
            logger.log_iteration(iteration, loss, policy_loss, value_loss, game_length)

            # Log MCTS stats
            num_nodes, max_depth = mcts.get_stats()
            logger.log_mcts_stats(iteration, num_nodes, max_depth)

            # Log time
            logger.log_time(iteration)

        print(f"Training complete for {game.__class__.__name__}!")

        # Save the trained model
        torch.save(network.state_dict(), f"trained_next_level_alphazero_{game.__class__.__name__.lower()}.pth")
        print(f"Model saved for {game.__class__.__name__}.")

        # Generate visualizations
        game_lengths = [len(history) for history in game_histories]
        plot_game_length_distribution(game_lengths, f'logs/game_length_distribution_{game.__class__.__name__.lower()}.png')

        action_frequencies = np.zeros(game.action_size)
        for history in game_histories:
            for _, action_probs, _ in history:
                action_frequencies += action_probs
        plot_action_heatmap(action_frequencies, int(np.sqrt(game.action_size)), f'logs/action_heatmap_{game.__class__.__name__.lower()}.png')

    # Run benchmarks
    run_benchmarks()

    # Shutdown Ray
    ray.shutdown()

    logger.close()

    print("All games trained and benchmarked successfully!")

if __name__ == "__main__":
    main()
