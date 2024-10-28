
import math
import numpy as np

class Node:
    def __init__(self, game, state, parent=None, action=None):
        self.game = game
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.ucb_score = float('inf')

class AdaptiveMCTS:
    def __init__(self, game, network, num_simulations=800, c_puct=1.0):
        self.game = game
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, state):
        root = Node(self.game, state)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            while node.children:
                node = self.select_child(node)
                search_path.append(node)

            value = self.evaluate_and_expand(node)
            self.backpropagate(search_path, value)

        return self.select_action(root)

    def select_child(self, node):
        return max(node.children, key=lambda c: c.ucb_score)

    def evaluate_and_expand(self, node):
        if self.game.is_terminal(node.state):
            return self.game.get_reward(node.state)

        policy, value = self.network.predict(node.state)
        valid_actions = self.game.get_valid_actions(node.state)
        policy = policy * valid_actions  # Mask invalid actions
        policy /= np.sum(policy)

        for action in range(len(policy)):
            if valid_actions[action]:
                child_state = self.game.get_next_state(node.state, action)
                child = Node(self.game, child_state, parent=node, action=action)
                node.children.append(child)

        return value

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visits += 1
            node.value += value
            if node.parent:
                node.ucb_score = self.ucb_score(node)
            value = -value  # Flip the value for the opponent

    def ucb_score(self, node):
        q_value = node.value / node.visits if node.visits > 0 else 0
        u_value = (self.c_puct * node.parent.policy[node.action] *
                   math.sqrt(node.parent.visits) / (1 + node.visits))
        return q_value + u_value

    def select_action(self, root):
        visits = [child.visits for child in root.children]
        return root.children[np.argmax(visits)].action

    def update_hyperparameters(self, game_phase):
        # Adapt exploration parameter based on game phase
        if game_phase == 'opening':
            self.c_puct = 1.5
        elif game_phase == 'midgame':
            self.c_puct = 1.0
        elif game_phase == 'endgame':
            self.c_puct = 0.5

if __name__ == "__main__":
    # Example usage
    from src.games.chess import Chess
    from src.alphazero.neural_network import NeuralNetwork

    game = Chess()
    network = NeuralNetwork(game)
    mcts = AdaptiveMCTS(game, network)

    state = game.get_initial_state()
    while not game.is_terminal(state):
        action = mcts.search(state)
        state = game.get_next_state(state, action)
        game_phase = game.get_game_phase(state)
        mcts.update_hyperparameters(game_phase)

    print("Game finished!")
