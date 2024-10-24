
import math
import numpy as np

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

    def search(self, state):
        for _ in range(self.num_simulations):
            self._simulate(state)

        s = self.game.get_state()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.action_size)]
        return counts

    def _simulate(self, state):
        s = self.game.get_state()
        if self.game.is_game_over():
            return -self.game.get_winner()

        if s not in self.Ps:
            self.Ps[s], v = self.model(state)
            self.Ps[s] = self.Ps[s].exp().detach().numpy()
            valid_moves = self.game.get_legal_moves()
            self.Ps[s] = self.Ps[s] * valid_moves  # mask invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valid_moves
                self.Ps[s] /= np.sum(self.Ps[s])
            self.Ns[s] = 0
            return -v

        best_u, best_a = -float('inf'), -1
        for a in range(self.game.action_size):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

            if u > best_u:
                best_u = u
                best_a = a

        a = best_a
        self.game.make_move(a)
        v = self._simulate(self.game.get_state())

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
