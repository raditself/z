
import numpy as np
from src.alphazero.mcts import AdaptiveMCTS
from src.alphazero.neural_network import DynamicNeuralNetwork

class BusinessEnvironment:
    def __init__(self):
        self.state_size = 10  # 10 business metrics
        self.action_size = 5  # 5 possible business strategies

    def get_initial_state(self):
        return np.random.rand(self.state_size)

    def get_next_state(self, state, action):
        # Simulate the effect of a business strategy on the metrics
        new_state = state.copy()
        new_state += np.random.normal(0, 0.1, self.state_size)  # Add some noise
        new_state[action] += 0.1  # The chosen strategy improves its corresponding metric
        return np.clip(new_state, 0, 1)  # Ensure values stay between 0 and 1

    def get_valid_actions(self, state):
        return np.ones(self.action_size)  # All actions are always valid

    def is_terminal(self, state):
        return np.max(state) >= 0.9  # Terminal when any metric reaches 90%

    def get_reward(self, state):
        return np.mean(state)  # Reward is the average of all metrics

class BusinessAlphaZero:
    def __init__(self):
        self.env = BusinessEnvironment()
        self.network = DynamicNeuralNetwork(self.env)
        self.mcts = AdaptiveMCTS(self.env, self.network)

    def train(self, num_iterations=1000):
        for i in range(num_iterations):
            state = self.env.get_initial_state()
            while not self.env.is_terminal(state):
                action = self.mcts.search(state)
                next_state = self.env.get_next_state(state, action)
                self.network.train(state, action, self.env.get_reward(next_state))
                state = next_state
            
            if i % 100 == 0:
                print(f"Iteration {i}, Average Reward: {self.evaluate()}")

    def evaluate(self, num_games=100):
        total_reward = 0
        for _ in range(num_games):
            state = self.env.get_initial_state()
            while not self.env.is_terminal(state):
                action = self.mcts.search(state)
                state = self.env.get_next_state(state, action)
            total_reward += self.env.get_reward(state)
        return total_reward / num_games

if __name__ == "__main__":
    business_ai = BusinessAlphaZero()
    business_ai.train()
    print(f"Final Average Reward: {business_ai.evaluate()}")
