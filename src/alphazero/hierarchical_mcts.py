

import math
import numpy as np

class HierarchicalMCTS:
    def __init__(self, game, net, args, external_knowledge_integrator):
        self.game = game
        self.net = net
        self.args = args
        self.external_knowledge_integrator = external_knowledge_integrator
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        self.hierarchical_levels = 3

    def search(self, state):
        for _ in range(self.args.num_mcts_sims):
            self._search_recursive(state, 0)
        
        s = self.game.string_representation(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.action_size)]
        return counts / np.sum(counts)

    def _search_recursive(self, state, level):
        if level == self.hierarchical_levels:
            return self._leaf_evaluation(state)

        s = self.game.string_representation(state)
        if s not in self.Ps:
            return self._leaf_evaluation(state)

        cur_best = -float('inf')
        best_act = -1

        # Integrate external knowledge
        external_knowledge = self.external_knowledge_integrator.integrate_knowledge(state)
        
        for a in range(self.game.action_size):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + self.args.epsilon)

            # Modify u based on external knowledge
            if 'move_probabilities' in external_knowledge:
                u += self.args.external_knowledge_weight * external_knowledge['move_probabilities'].get(a, 0)

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_s = self.game.get_next_state(state, a)
        v = self._search_recursive(next_s, level + 1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def _leaf_evaluation(self, state):
        s = self.game.string_representation(state)
        if s not in self.Ps:
            self.Ps[s], v = self.net(self.game.get_encoded_state(state))
            self.Ps[s] = self.Ps[s].detach().cpu().numpy()
            v = v.item()
            valids = self.game.get_valid_moves(state)
            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                self.Ps[s] = valids / np.sum(valids)
            self.Vs[s] = valids
            self.Ns[s] = 0
            
            # Integrate external knowledge
            external_knowledge = self.external_knowledge_integrator.integrate_knowledge(state)
            if 'value' in external_knowledge:
                v = (v + external_knowledge['value']) / 2  # Average with external value
            
            return v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # Dynamic branching factor based on position complexity
        complexity = self._calculate_complexity(state)
        branching_factor = max(5, min(20, int(complexity * self.game.action_size)))

        # Integrate external knowledge
        external_knowledge = self.external_knowledge_integrator.integrate_knowledge(state)

        for a in np.argsort(self.Ps[s])[-branching_factor:]:
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + self.args.epsilon)

                # Modify u based on external knowledge
                if 'move_probabilities' in external_knowledge:
                    u += self.args.external_knowledge_weight * external_knowledge['move_probabilities'].get(a, 0)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.game.get_next_state(state, a)
        v = self._search_recursive(next_s, 0)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def _calculate_complexity(self, state):
        # Implement complexity calculation logic here
        # This is a placeholder implementation
        return 0.5

