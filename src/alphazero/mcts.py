
import numpy as np

class MCTS:
    def __init__(self, game, model, num_simulations=100):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.transposition_table = {}

    def search(self, state):
        self.transposition_table.clear()  # Clear the table before each search
        for _ in range(self.num_simulations):
            self._simulate(state, depth=0)

    def _simulate(self, state, depth):
        state_hash = self.game.get_state_hash(state)
        
        if state_hash in self.transposition_table:
            return self.transposition_table[state_hash]

        if depth > 10 or self.game.is_terminal(state):
            value = -self.game.get_reward(state)
            self.transposition_table[state_hash] = value
            return value

        policy, value = self.model.predict(state)
        action = np.argmax(policy)
        next_state = self.game.get_next_state(state, action)
        
        value = -self._simulate(next_state, depth + 1)
        self.transposition_table[state_hash] = value
        return value

    def get_action_prob(self, state):
        self.search(state)
        # Dummy implementation: return uniform distribution over actions
        return np.ones(self.game.get_action_size()) / self.game.get_action_size()

    def clear_transposition_table(self):
        self.transposition_table.clear()

