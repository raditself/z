
import math
import numpy as np
import torch

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.device = next(model.parameters()).device

    def search(self, states):
        if not isinstance(states, list):
            states = [states]
        
        batch_size = len(states)
        roots = [Node(0) for _ in range(batch_size)]
        for i, root in enumerate(roots):
            root.state = states[i]

        for _ in range(self.args.num_simulations):
            nodes = roots
            search_paths = [[node] for node in nodes]

            while all(node.expanded() for node in nodes):
                actions = []
                next_nodes = []
                for node in nodes:
                    action, next_node = self.select_child(node)
                    actions.append(action)
                    next_nodes.append(next_node)
                nodes = next_nodes
                for i, node in enumerate(nodes):
                    search_paths[i].append(node)

            # Expand and evaluate
            expand_states = []
            expand_indices = []
            for i, node in enumerate(nodes):
                if not node.expanded():
                    parent = search_paths[i][-2]
                    state = parent.state
                    action = actions[i]
                    next_state, _ = self.game.get_next_state(state, action)
                    expand_states.append(next_state)
                    expand_indices.append(i)

            if expand_states:
                expand_states = torch.stack([torch.from_numpy(state).float() for state in expand_states]).to(self.device)
                policies, values = self.model(expand_states)
                policies = policies.cpu().numpy()
                values = values.cpu().numpy()

                for idx, i in enumerate(expand_indices):
                    node = nodes[i]
                    state = expand_states[idx].cpu().numpy()
                    policy = self.game.get_valid_moves(state) * policies[idx]
                    policy /= np.sum(policy)
                    value = values[idx][0]

                    node.state = state
                    for action, p in enumerate(policy):
                        if p > 0:
                            node.children[action] = Node(p)
                    
                    self.backpropagate(search_paths[i], value)
            else:
                for i, node in enumerate(nodes):
                    value = self.game.get_reward(node.state)
                    self.backpropagate(search_paths[i], value)

        return [self.select_action(root) for root in roots]

    def select_child(self, node):
        _, action, child = max((self.ucb_score(node, child), action, child) for action, child in node.children.items())
        return action, child

    def ucb_score(self, parent, child):
        prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        value_score = -child.value()
        return value_score + self.args.c_puct * prior_score

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1

    def select_action(self, root):
        visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
        if len(visit_counts) == 0:
            return None
        _, action = max(((c, a) for a, c in visit_counts), key=lambda x: x[0])
        return action
