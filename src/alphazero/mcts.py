import numpy as np

class MCTS:
    def __init__(self, game, model, num_simulations=100):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations

    def search(self, state):
        for _ in range(self.num_simulations):
            self._simulate(state, depth=0)

    def _simulate(self, state, depth):
        if depth > 10 or self.game.is_terminal(state):  # Add a depth limit
            return -self.game.get_reward(state)

        policy, value = self.model.predict(state)
        action = np.argmax(policy)
        next_state = self.game.get_next_state(state, action)
        
        return -self._simulate(next_state, depth + 1)

    def get_action_prob(self, state):
        self.search(state)
        # Dummy implementation: return uniform distribution over actions
        return np.ones(self.game.get_action_size()) / self.game.get_action_size()
