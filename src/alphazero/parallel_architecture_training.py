
import os
import torch
import torch.multiprocessing as mp
import logging
import matplotlib.pyplot as plt
from typing import List, Dict
from src.alphazero.alphazero import AlphaZero
from src.alphazero.model import AlphaZeroNetwork
from src.alphazero.self_play import self_play
from src.alphazero.train import train_network

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ParallelArchitectureTrainer:
    def __init__(self, game, architectures: List[Dict], num_actors: int, num_gpus: int, initial_lr: float = 0.001):
        self.game = game
        self.architectures = architectures
        self.num_actors = num_actors
        self.num_gpus = num_gpus
        self.models = []
        self.optimizers = []
        self.performance_history = {i: [] for i in range(len(architectures))}
        self.initial_lr = initial_lr
        self.lr_decay_factor = 0.5
        self.lr_decay_patience = 5
        logging.info(f"Initialized ParallelArchitectureTrainer with {len(architectures)} architectures, {num_actors} actors, and {num_gpus} GPUs")

    def visualize_performance(self):
        plt.figure(figsize=(12, 8))
        for i, performances in self.performance_history.items():
            plt.plot(performances, label=f"Model {i}")
        plt.xlabel("Iteration")
        plt.ylabel("Performance")
        plt.title("Model Performance over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig("performance_plot.png")
        plt.close()
        logging.info("Performance plot saved as performance_plot.png")

    def visualize_time_per_iteration(self, times):
        plt.figure(figsize=(12, 8))
        plt.plot(times)
        plt.xlabel("Iteration")
        plt.ylabel("Time (seconds)")
        plt.title("Time per Iteration")
        plt.grid(True)
        plt.savefig("time_per_iteration_plot.png")
        plt.close()
        logging.info("Time per iteration plot saved as time_per_iteration_plot.png")

    def visualize_memory_usage(self, memory_usages):
        plt.figure(figsize=(12, 8))
        plt.plot(memory_usages)
        plt.xlabel("Iteration")
        plt.ylabel("Memory Usage (%)")
        plt.title("Memory Usage over Time")
        plt.grid(True)
        plt.savefig("memory_usage_plot.png")
        plt.close()
        logging.info("Memory usage plot saved as memory_usage_plot.png")

    def adjust_learning_rate(self, optimizer, iteration, best_performance):
        if iteration > 0 and iteration % self.lr_decay_patience == 0:
            if self.performance_history[best_performance][-1] <= self.performance_history[best_performance][-self.lr_decay_patience]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.lr_decay_factor
                logging.info(f"Learning rate adjusted to {param_group['lr']}")

    def save_performance_history(self, filename: str):
        torch.save(self.performance_history, filename)
        logging.info(f"Performance history saved to {filename}")

    def load_performance_history(self, filename: str):
        self.performance_history = torch.load(filename)
        logging.info(f"Performance history loaded from {filename}")

    def initialize_models(self):
        for arch in self.architectures:
            model = AlphaZeroNetwork(self.game, **arch)
            optimizer = torch.optim.Adam(model.parameters())
            self.models.append(model)
            self.optimizers.append(optimizer)

    def distribute_actors(self):
        actors_per_gpu = self.num_actors // self.num_gpus
        return [actors_per_gpu] * self.num_gpus + [self.num_actors % self.num_gpus]

    def add_architecture(self, architecture: Dict):
        model = AlphaZeroNetwork(self.game, **architecture)
        optimizer = torch.optim.Adam(model.parameters())
        self.models.append(model)
        self.optimizers.append(optimizer)
        self.architectures.append(architecture)
        self.performance_history[len(self.models) - 1] = []
        logging.info(f"Added new architecture: {architecture}")

    def remove_architecture(self, index: int):
        if 0 <= index < len(self.models):
            del self.models[index]
            del self.optimizers[index]
            del self.architectures[index]
            del self.performance_history[index]
            logging.info(f"Removed architecture at index {index}")
        else:
            logging.warning(f"Invalid index {index} for removing architecture")

    def dynamic_architecture_adjustment(self, performances: Dict[int, float], threshold: float = 0.8):
        best_performance = max(performances.values())
        worst_performance = min(performances.values())
        performance_range = best_performance - worst_performance

        # Remove underperforming architectures
        indices_to_remove = [i for i, perf in performances.items() if perf < worst_performance + threshold * performance_range]
        for index in sorted(indices_to_remove, reverse=True):
            self.remove_architecture(index)

        # Add new architecture based on best performing one
        if len(self.models) > 0:
            best_model_index = max(performances, key=performances.get)
            best_architecture = self.architectures[best_model_index]
            new_architecture = self.mutate_architecture(best_architecture)
            self.add_architecture(new_architecture)

    def mutate_architecture(self, architecture: Dict) -> Dict:
        import random
        new_architecture = architecture.copy()
        mutation_factor = 0.1  # 10% mutation

        for key in new_architecture:
            if isinstance(new_architecture[key], int):
                new_architecture[key] = max(1, int(new_architecture[key] * (1 + random.uniform(-mutation_factor, mutation_factor))))
            elif isinstance(new_architecture[key], float):
                new_architecture[key] = max(0.0001, new_architecture[key] * (1 + random.uniform(-mutation_factor, mutation_factor)))

        return new_architecture

    def train_parallel(self, num_iterations: int):
        self.initialize_models()
        mp.set_start_method('spawn')
        logging.info(f"Starting parallel training for {num_iterations} iterations")

        best_performance = float('-inf')
        iterations_without_improvement = 0
        max_iterations_without_improvement = 10

        import time
        import psutil

        iteration_times = []
        memory_usages = []

        for iteration in range(num_iterations):
            start_time = time.time()
            logging.info(f"Starting iteration {iteration + 1}/{num_iterations}")
            processes = []
            for model_idx, model in enumerate(self.models):
                p = mp.Process(target=self.train_single_architecture, args=(model_idx, model, iteration))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            try:
                best_model, performances = self.compare_architectures()
                end_time = time.time()
                iteration_time = end_time - start_time
                memory_usage = psutil.virtual_memory().percent

                iteration_times.append(iteration_time)
                memory_usages.append(memory_usage)

                logging.info(f"Iteration {iteration + 1} completed in {iteration_time:.2f} seconds.")
                logging.info(f"Memory usage: {memory_usage:.2f}%")
                logging.info(f"Best model: {best_model}, Performance: {performances[best_model]:.4f}")

                # Update performance history
                for model_idx, performance in performances.items():
                    self.performance_history[model_idx].append(performance)

                # Adjust learning rate
                for model_idx, optimizer in enumerate(self.optimizers):
                    self.adjust_learning_rate(optimizer, iteration, model_idx)

                # Dynamic architecture adjustment
                self.dynamic_architecture_adjustment(performances)

                if performances[best_model] > best_performance:
                    best_performance = performances[best_model]
                    iterations_without_improvement = 0
                    self.save_training_state(iteration, best_model, best_performance)
                else:
                    iterations_without_improvement += 1

                if iterations_without_improvement >= max_iterations_without_improvement:
                    logging.info(f"Early stopping after {iteration + 1} iterations due to lack of improvement")
                    break

                # Visualize performance after each iteration
                self.visualize_performance()
                self.visualize_time_per_iteration(iteration_times)
                self.visualize_memory_usage(memory_usages)

                # Save performance history periodically
                if iteration % 10 == 0:
                    self.save_performance_history(f"performance_history_iteration_{iteration}.pth")

            except Exception as e:
                logging.error(f"Error during iteration {iteration + 1}: {str(e)}")

        # Final performance visualization and history save
        self.visualize_performance()
        self.visualize_time_per_iteration(iteration_times)
        self.visualize_memory_usage(memory_usages)
        self.save_performance_history("final_performance_history.pth")

    def round_robin_tournament(self):
        logging.info("Starting round-robin tournament...")
        results = {i: 0 for i in range(len(self.models))}
        games_played = {i: 0 for i in range(len(self.models))}

        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                logging.info(f"Playing game: Model {i} vs Model {j}")
                score = self.play_game(self.models[i], self.models[j])
                if score > 0:
                    results[i] += 1
                elif score < 0:
                    results[j] += 1
                else:
                    results[i] += 0.5
                    results[j] += 0.5
                games_played[i] += 1
                games_played[j] += 1

        # Normalize scores
        for i in results:
            results[i] = results[i] / games_played[i] if games_played[i] > 0 else 0

        return results

    def compare_architectures(self):
        logging.info("Comparing architectures...")
        
        results = self.round_robin_tournament()
        
        # Log results
        for i, score in results.items():
            logging.info(f"Model {i}: {score:.2f} average points")
        
        # Determine the best model
        best_model = max(results, key=results.get)
        logging.info(f"Best model: {best_model}, Performance: {results[best_model]:.2f}")
        
        return best_model, results

    def play_game(self, model1, model2):
        # Create AlphaZero instances for both models
        alphazero1 = AlphaZero(self.game, model1)
        alphazero2 = AlphaZero(self.game, model2)

        # Initialize the game
        state = self.game.get_initial_state()
        done = False
        player = 1

        while not done:
            if player == 1:
                action, _ = alphazero1.get_action(state)
            else:
                action, _ = alphazero2.get_action(state)

            state = self.game.get_next_state(state, action, player)
            done = self.game.is_terminal(state)
            reward = self.game.get_reward(state, player)
            player = self.game.get_opponent(player)

        if reward == 1:
            logging.info("Model 1 wins")
            return 1  # model1 wins
        elif reward == -1:
            logging.info("Model 2 wins")
            return -1  # model2 wins
        else:
            logging.info("Game ended in a draw")
            return 0  # draw

# Usage example
# game = YourGameImplementation()
# architectures = [
#     {"num_resblocks": 19, "num_hidden": 256},
#     {"num_resblocks": 39, "num_hidden": 256},
#     {"num_resblocks": 19, "num_hidden": 512},
# ]
# trainer = ParallelArchitectureTrainer(game, architectures, num_actors=116, num_gpus=4)
# trainer.train_parallel(num_iterations=200)
