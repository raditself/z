
import math
import numpy as np

class MCTSNode:
    def __init__(self, game_state, parent=None, action=None):
        self.game_state = game_state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.prior = 0

class MCTS:
    def __init__(self, game, model, num_simulations=800, c_puct=1.0):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, root_state):
        root = MCTSNode(root_state)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.children:
                node = self.select_child(node)
                search_path.append(node)
            
            # Expansion and evaluation
            value = self.expand_and_evaluate(node)
            
            # Backpropagation
            self.backpropagate(search_path, value)
        
        return self.get_action_probs(root)

    def select_child(self, node):
        best_score = -float('inf')
        best_child = None
        
        for child in node.children:
            ucb_score = child.value / (child.visits + 1e-8) +                         self.c_puct * child.prior * math.sqrt(node.visits) / (1 + child.visits)
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child

    def expand_and_evaluate(self, node):
        game_state = node.game_state
        
        if self.game.is_game_over(game_state):
            return -self.game.get_winner(game_state)
        
        # Use the neural network to predict action probabilities and value
        action_probs, value = self.model.predict(game_state)
        
        valid_moves = self.game.get_valid_moves(game_state)
        action_probs = action_probs * valid_moves  # Mask invalid moves
        action_probs /= np.sum(action_probs)
        
        for action, prob in enumerate(action_probs):
            if valid_moves[action]:
                next_state = self.game.get_next_state(game_state, action)
                child = MCTSNode(next_state, parent=node, action=action)
                child.prior = prob
                node.children.append(child)
        
        return value

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visits += 1
            node.value += value
            value = -value  # Switch perspective

    def get_action_probs(self, root, temperature=1.0):
        visits = [child.visits for child in root.children]
        actions = [child.action for child in root.children]
        
        if temperature == 0:
            best_action = actions[np.argmax(visits)]
            probs = [0] * len(actions)
            probs[actions.index(best_action)] = 1
            return probs
        
        visits = [v ** (1 / temperature) for v in visits]
        total_visits = sum(visits)
        probs = [v / total_visits for v in visits]
        
        return probs

# Usage example:
# game = YourGameClass()
# model = YourNeuralNetworkModel()
# mcts = MCTS(game, model)
# best_action = mcts.search(initial_state)
