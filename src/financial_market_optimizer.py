
import numpy as np
from src.alphazero.mcts import MCTS
from src.alphazero.neural_network import NeuralNetwork

class FinancialMarketOptimizer:
    def __init__(self, initial_capital, num_assets):
        self.initial_capital = initial_capital
        self.num_assets = num_assets
        self.nn = NeuralNetwork(input_shape=(num_assets,), output_shape=(num_assets,))
        self.mcts = MCTS(self.nn)

    def optimize_portfolio(self, market_data):
        state = self.preprocess_market_data(market_data)
        action = self.mcts.search(state)
        return self.postprocess_action(action)

    def preprocess_market_data(self, market_data):
        # Convert market data to a format suitable for the neural network
        return np.array(market_data)

    def postprocess_action(self, action):
        # Convert MCTS action to portfolio allocation
        return action / np.sum(action) * self.initial_capital

def main():
    optimizer = FinancialMarketOptimizer(initial_capital=100000, num_assets=5)
    market_data = [1.0, 1.2, 0.8, 1.1, 0.9]  # Example market data
    optimal_allocation = optimizer.optimize_portfolio(market_data)
    print(f"Optimal portfolio allocation: {optimal_allocation}")

if __name__ == "__main__":
    main()
