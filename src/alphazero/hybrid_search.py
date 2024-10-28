
import numpy as np
from src.alphazero.mcts import AdaptiveMCTS

class AlphaBetaNode:
    def __init__(self, game, state, player):
        self.game = game
        self.state = state
        self.player = player

    def alpha_beta_search(self, depth, alpha, beta):
        if depth == 0 or self.game.is_terminal(self.state):
            return self.game.get_reward(self.state, self.player)

        if self.player == 0:  # Maximizing player
            value = float('-inf')
            for action in self.game.get_valid_actions(self.state):
                child_state = self.game.get_next_state(self.state, action)
                child_node = AlphaBetaNode(self.game, child_state, 1 - self.player)
                value = max(value, child_node.alpha_beta_search(depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:  # Minimizing player
            value = float('inf')
            for action in self.game.get_valid_actions(self.state):
                child_state = self.game.get_next_state(self.state, action)
                child_node = AlphaBetaNode(self.game, child_state, 1 - self.player)
                value = min(value, child_node.alpha_beta_search(depth - 1, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

class HybridSearch(AdaptiveMCTS):
    def __init__(self, game, network, num_simulations=800, alpha_beta_depth=3):
        super().__init__(game, network, num_simulations)
        self.alpha_beta_depth = alpha_beta_depth

    def search(self, state):
        if self.game.get_game_phase(state) == 'endgame':
            return self.alpha_beta_search(state)
        else:
            return super().search(state)

    def alpha_beta_search(self, state):
        root = AlphaBetaNode(self.game, state, 0)
        best_value = float('-inf')
        best_action = None

        for action in self.game.get_valid_actions(state):
            child_state = self.game.get_next_state(state, action)
            child_node = AlphaBetaNode(self.game, child_state, 1)
            value = child_node.alpha_beta_search(self.alpha_beta_depth - 1, float('-inf'), float('inf'))
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

if __name__ == "__main__":
    from src.games.chess import Chess
    from src.alphazero.neural_network import DynamicNeuralNetwork

    game = Chess()
    network = DynamicNeuralNetwork(game)
    hybrid_search = HybridSearch(game, network)

    state = game.get_initial_state()
    while not game.is_terminal(state):
        action = hybrid_search.search(state)
        state = game.get_next_state(state, action)
        print(f"Action taken: {action}")

    print("Game finished!")
