
import math
import numpy as np

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

    def search(self, state):
        root = Node(0)
        root.state = state
        for _ in range(self.args.num_simulations):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
            next_state, _ = self.game.get_next_state(state, action)
            value = self.evaluate(next_state, search_path)

            self.backpropagate(search_path, value)

        return self.select_action(root)

    def select_child(self, node):
        _, action, child = max((self.ucb_score(node, child), action, child) for action, child in node.children.items())
        return action, child

    def ucb_score(self, parent, child):
        prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        value_score = -child.value()
        return value_score + self.args.c_puct * prior_score

    def evaluate(self, state, search_path):
        value = self.game.get_reward(state)
        if value is None:
            policy, value = self.model(state)
            policy = self.game.get_valid_moves(state) * policy
            policy /= np.sum(policy)
            for action, p in enumerate(policy):
                if p > 0:
                    search_path[-1].children[action] = Node(p)
        return value

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
