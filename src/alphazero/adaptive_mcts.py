
import math
import numpy as np

class AdaptiveMCTS:
    def __init__(self, game, net, args, opponent_model):
        self.game = game
        self.net = net
        self.args = args
        self.opponent_model = opponent_model
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}

    def search(self, state, game_phase):
        s = self.game.string_representation(state)
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(state, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            self.Ps[s], v = self.net(
                self.game.get_encoded_state(state).unsqueeze(0),
                torch.tensor([game_phase]).unsqueeze(0)
            )
            self.Ps[s] = self.Ps[s].detach().cpu().numpy().squeeze()
            valids = self.game.get_valid_moves(state)
            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v.item()

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + self.args.epsilon)

                # Incorporate opponent modeling
                opponent_tendency = self.opponent_model.get_move_probability(state, a)
                u += self.args.opponent_exploit_factor * (1 - opponent_tendency)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.get_next_state(state, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s, game_phase)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def get_action_prob(self, state, game_phase, temp=1):
        for _ in range(self.args.num_mcts_sims):
            self.search(state, game_phase)

        s = self.game.string_representation(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.get_action_size())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def calculate_game_phase(self, state):
        # Implement game phase calculation logic here
        # This could be based on the number of moves played, pieces on the board, etc.
        # Return a value between 0 and 1 representing the game phase
        pass
