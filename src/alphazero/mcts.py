
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
        self.action_size = game.action_size
        self.state_size = game.state_size
        self.num_processes = num_processes

        # Use numpy arrays for faster access
        self.Qsa = np.zeros((self.state_size, self.action_size))
        self.Nsa = np.zeros((self.state_size, self.action_size), dtype=np.int32)
        self.Ns = np.zeros(self.state_size, dtype=np.int32)
        self.Ps = np.zeros((self.state_size, self.action_size))

        self.Es = np.zeros(self.state_size, dtype=np.int8)
        self.Vs = np.zeros((self.state_size, self.action_size), dtype=np.int8)

        self.transposition_table = {}

    def search(self, state):
        s = self.game.get_state()
        
        with mp.Pool(processes=self.num_processes) as pool:
            results = pool.map(partial(self._search, s=s), range(self.num_simulations))
        
        for result in results:
            self._update_stats(result)
        
        counts = self.Nsa[s]
        return counts

    def _search(self, _, s):
        if s in self.transposition_table:
            return self.transposition_table[s], s, None, None

        if self.Es[s] == 0:
            self.Es[s] = self.game.is_game_over()
        if self.Es[s] != 0:
            return -self.Es[s], s, None, None

        if self.Ns[s] == 0:
            state = self.game.get_state()
            self.Ps[s], v = self.model(state)
            self.Ps[s] = np.exp(self.Ps[s])
            valids = self.game.get_legal_moves()
            self.Ps[s] *= valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            
            self.Vs[s] = valids
            return -v, s, None, None

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.action_size):
            if valids[a]:
                if self.Nsa[s][a] > 0:
                    u = self.Qsa[s][a] + self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[s][a])
                else:
                    u = self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.game.get_next_state(a)
        self.game.make_move(a)

        v, _, _, _ = self._search(None, next_s)

        self.transposition_table[s] = -v
        return -v, s, a, v

    def _update_stats(self, result):
        v, s, a, _ = result
        if a is not None:
            self.Qsa[s][a] = (self.Nsa[s][a] * self.Qsa[s][a] + v) / (self.Nsa[s][a] + 1)
            self.Nsa[s][a] += 1
            self.Ns[s] += 1

    def get_action_prob(self, state, temp=1):
        counts = self.search(state)
        if temp == 0:
            best_a = np.argmax(counts)
            probs = np.zeros(self.action_size)
            probs[best_a] = 1
            return probs
        else:
            counts = [x ** (1. / temp) for x in counts]
            total = sum(counts)
            probs = [x / total for x in counts]
            return probs
