
import math
import numpy as np
import multiprocessing as mp
from functools import partial

class MCTS:
    def __init__(self, game, model, num_simulations=800, c_puct=1.0, num_processes=4):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.num_processes = num_processes
        self.action_size = game.action_size
        self.transposition_table = {}

    def search(self, state):
        if self.num_processes > 1:
            with mp.Pool(self.num_processes) as pool:
                results = pool.map(partial(self._simulate, state=state), range(self.num_simulations))
        else:
            results = [self._simulate(state) for _ in range(self.num_simulations)]

        s = self.game.get_state()
        counts = np.bincount(results, minlength=self.action_size)
        return counts

    def _simulate(self, state):
        game_copy = self.game.clone()
        s = game_copy.get_state()

        if s in self.transposition_table:
            return self.transposition_table[s]

        if game_copy.is_game_over():
            return -game_copy.get_winner()

        policy, v = self.model(state)
        policy = policy.exp().detach().numpy()
        valid_moves = game_copy.get_legal_moves()
        policy = policy * valid_moves  # mask invalid moves
        policy_sum = np.sum(policy)

        if policy_sum > 0:
            policy /= policy_sum  # renormalize
        else:
            policy = valid_moves / np.sum(valid_moves)

        best_a = np.argmax(policy + np.random.randn(self.action_size) * 1e-5)
        game_copy.make_move(best_a)
        v = self._simulate(game_copy.get_state())

        self.transposition_table[s] = best_a
        return -v

    def get_action_prob(self, state, temp=1):
        counts = self.search(state)
        if temp == 0:
            best_a = np.argmax(counts)
            probs = np.zeros(self.action_size)
            probs[best_a] = 1
            return probs
        else:
            counts = [x ** (1. / temp) for x in counts]
            probs = [x / float(sum(counts)) for x in counts]
            return probs
