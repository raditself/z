
import math
import numpy as np
import multiprocessing as mp
from functools import partial

class MCTS:
    def __init__(self, game, model, num_simulations=800, c_puct=1.0):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.Qsa = {}  # stores Q values for s,a
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}   # stores #times board s was visited
        self.Ps = {}   # stores initial policy (returned by neural net)

        self.Es = {}   # stores game.get_game_ended ended for board s
        self.Vs = {}   # stores game.get_valid_moves for board s

    def search(self, state):
        s = self.game.get_state()
        for _ in range(self.num_simulations):
            self._search(s)
        
        s = self.game.get_state()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.action_size)]
        return counts

    def _search(self, s):
        if s not in self.Es:
            self.Es[s] = self.game.is_game_over()
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            self.Ps[s], v = self.model(self.game.get_state())
            self.Ps[s] = self.Ps[s].exp().detach().numpy()
            valids = self.game.get_legal_moves()
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.action_size):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.game.get_next_state(a)
        self.game.make_move(a)

        v = self._search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
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
