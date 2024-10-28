
import torch
import torch.optim as optim
from src.alphazero.mcts import AdaptiveMCTS
from src.alphazero.neural_network import DynamicNeuralNetwork, train_network
from src.games.chess import Chess
from src.alphazero.self_play import self_play
from src.alphazero.evaluate import evaluate_against_base
from tqdm import tqdm

def train_alphazero(game, num_iterations=100, num_episodes=100, num_epochs=10, batch_size=32):
    network = DynamicNeuralNetwork(game)
    optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-4)
    mcts = AdaptiveMCTS(game, network)

    for iteration in tqdm(range(num_iterations), desc="Training AlphaZero"):
        # Self-play
        game_data = self_play(game, mcts, network, num_episodes)

        # Train network
        for _ in range(num_epochs):
            train_network(network, optimizer, game_data)

        # Update MCTS
        mcts.update_hyperparameters(game.get_game_phase(game.get_initial_state()))

        # Evaluate
        if iteration % 10 == 0:
            base_mcts = AdaptiveMCTS(game, None)  # MCTS without neural network
            win_rate = evaluate_against_base(game, mcts, network, base_mcts)
            print(f"Iteration {iteration}, Win rate against base MCTS: {win_rate:.2f}")

    return network

if __name__ == "__main__":
    game = Chess()
    trained_network = train_alphazero(game)
    torch.save(trained_network.state_dict(), "trained_alphazero_chess.pth")
    print("Training complete. Model saved.")
