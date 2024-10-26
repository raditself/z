import numpy as np
import math

class MCTS:
    def __init__(self, game, model, num_simulations=100, c_puct=1.0, c_pw=0.5, alpha_pw=0.5):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.c_pw = c_pw  # Progressive widening constant
        self.alpha_pw = alpha_pw  # Progressive widening exponent
        self.transposition_table = {}
        self.visit_counts = {}
        self.Q_values = {}
        self.P_values = {}

    def search(self, state):
        for _ in range(self.num_simulations):
            self._simulate(state)

    def _simulate(self, state):
        if self.game.is_terminal(state):
            return -self.game.get_reward(state)

        state_hash = hash(str(state))
        if state_hash not in self.visit_counts:
            self.visit_counts[state_hash] = 0
            self.Q_values[state_hash] = {}
            policy, value = self.model.predict(state)
            self.P_values[state_hash] = policy
            return -value

        N = self.visit_counts[state_hash]
        max_actions = math.ceil(self.c_pw * (N ** self.alpha_pw))  # Progressive widening
        legal_actions = self.game.get_legal_actions(state)
        explored_actions = list(self.Q_values[state_hash].keys())

        if len(explored_actions) < min(max_actions, len(legal_actions)):
            unexplored_actions = [a for a in legal_actions if a not in explored_actions]
            action = np.random.choice(unexplored_actions)
        else:
            action = max(explored_actions, key=lambda a: self._ucb_score(state_hash, a))

        next_state = self.game.get_next_state(state, action)
        value = self._simulate(next_state)

        self.visit_counts[state_hash] += 1
        if action not in self.Q_values[state_hash]:
            self.Q_values[state_hash][action] = 0

        self.Q_values[state_hash][action] += (value - self.Q_values[state_hash][action]) / self.visit_counts[state_hash]

        return -value

    def _ucb_score(self, state_hash, action):
        Q = self.Q_values[state_hash].get(action, 0)
        P = self.P_values[state_hash][action]
        N = self.visit_counts[state_hash]
        n = sum(1 for a in self.Q_values[state_hash] if a == action)
        return Q + self.c_puct * P * math.sqrt(N) / (1 + n)

    def get_action_prob(self, state):
        self.search(state)
        state_hash = hash(str(state))
        counts = np.array([self.Q_values[state_hash].get(a, 0) for a in range(self.game.get_action_size())])
        return counts / np.sum(counts)

