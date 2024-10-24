
import math
import numpy as np
from multiprocessing import Pool, cpu_count
from .opening_book import OpeningBook

class MCTS:
    def __init__(self, game, model, args, opening_book=None):
        self.game = game
        self.model = model
        self.args = args
        self.opening_book = opening_book or OpeningBook()

    def search(self, state):
        # Check opening book first
        chess_board = self.game.to_chess_board()
        book_move = self.opening_book.get_move(chess_board)
        if book_move:
            return self.chess_move_to_action(book_move), None

        root = Node(self.game, state)
        
        # Parallelize the search process
        with Pool(processes=cpu_count()) as pool:
            for _ in range(self.args.num_simulations // cpu_count()):
                leaf_nodes = pool.map(self.simulate, [root] * cpu_count())
                
                for leaf in leaf_nodes:
                    self.backpropagate(leaf)

        return self.get_action_probs(root)

    def simulate(self, node):
        while not node.is_leaf():
            node = node.select_child()
        
        value = self.game.get_game_ended(node.state)
        if value is None:
            node.expand(self.model)
            value = node.value
        
        return node

    def backpropagate(self, node):
        while node is not None:
            node.update(node.value)
            node = node.parent

    def get_action_probs(self, root, temperature=1):
        visits = [child.visit_count for child in root.children]
        actions = [child.action for child in root.children]
        if temperature == 0:
            action_idx = np.argmax(visits)
            action_probs = [0] * len(actions)
            action_probs[action_idx] = 1
            return actions, action_probs
        
        visits = [x ** (1. / temperature) for x in visits]
        total = sum(visits)
        action_probs = [x / total for x in visits]
        return actions, action_probs

    def chess_move_to_action(self, chess_move):
        from_square = chess_move.from_square
        to_square = chess_move.to_square
        from_row, from_col = 7 - (from_square // 8), from_square % 8
        to_row, to_col = 7 - (to_square // 8), to_square % 8
        return from_row * 64 + from_col * 8 + to_row * 8 + to_col

class Node:
    def __init__(self, game, state, parent=None, action=None):
        self.game = game
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visit_count = 0
        self.value = 0
        self.prior = 0

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        c_puct = 1.0
        
        def ucb_score(child):
            u = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            return child.value + u

        return max(self.children, key=ucb_score)

    def expand(self, model):
        policy, value = model.predict(self.state)
        valid_moves = self.game.get_valid_moves(self.state)
        policy = policy * valid_moves  # mask invalid moves
        policy /= np.sum(policy)
        
        for action, prob in enumerate(policy):
            if valid_moves[action]:
                child_state = self.game.get_next_state(self.state, action)
                child = Node(self.game, child_state, parent=self, action=action)
                child.prior = prob
                self.children.append(child)
        
        self.value = value

    def update(self, value):
        self.visit_count += 1
        self.value += (value - self.value) / self.visit_count

