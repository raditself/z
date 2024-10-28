
import numpy as np
import torch
from src.alphazero.model import AlphaZeroNetwork
from src.alphazero.game import GameState
from src.alphazero.mcts import MCTS

class OnlineLearningSystem:
    def __init__(self, model: AlphaZeroNetwork, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def update_from_game(self, game_state: GameState, move_probabilities: np.ndarray, outcome: float):
        # Convert game state to model input
        state_input = self.model.convert_to_input(game_state)

        # Get model predictions
        policy, value = self.model(state_input)

        # Calculate losses
        policy_loss = -torch.sum(torch.tensor(move_probabilities) * torch.log(policy))
        value_loss = (value - outcome) ** 2

        # Combine losses
        total_loss = policy_loss + value_loss

        # Backpropagate and update model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def learn_from_self_play(self, num_games: int = 100):
        for _ in range(num_games):
            game = GameState()
            mcts = MCTS(self.model)

            while not game.is_game_over():
                for _ in range(800):  # Number of MCTS simulations
                    mcts.search(game)

                policy = mcts.get_policy(game)
                action = np.random.choice(len(policy), p=policy)
                game = game.make_move(action)

            outcome = game.get_outcome()

            # Update model based on the game
            for state, policy in zip(game.state_history, game.policy_history):
                self.update_from_game(state, policy, outcome)

    def learn_from_human_game(self, game_history: list[GameState], move_history: list[int], outcome: float):
        for state, move in zip(game_history, move_history):
            # Create a one-hot encoded move probability
            move_prob = np.zeros(state.action_size())
            move_prob[move] = 1.0

            self.update_from_game(state, move_prob, outcome)

# Add this to the end of the file to allow importing the class
if __name__ == "__main__":
    print("Online Learning System module loaded.")
