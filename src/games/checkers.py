
import numpy as np
import torch
from src.alphazero.game import Game

class CheckersGame(Game):
    def __init__(self):
        self.board_size = 8
        self.action_size = self.board_size * self.board_size
        self.complexity = 1.0  # Default complexity
        self.initial_pieces = 12 * 2  # 12 pieces per player

    def get_initial_state(self):
        # Simplified initial state: 1 for player 1, -1 for player 2, 0 for empty
        board = np.zeros((self.board_size, self.board_size))
        num_rows = max(1, int(3 * self.complexity))  # Adjust number of rows based on complexity
        board[:num_rows, ::2] = 1
        board[-num_rows:, 1::2] = -1
        return board

    def get_next_state(self, state, action):
        # Simplified: just flip a random piece
        next_state = state.copy()
        x, y = action // self.board_size, action % self.board_size
        next_state[x, y] *= -1
        return next_state

    def get_valid_moves(self, state):
        return (state == 0).flatten()

    def is_terminal(self, state):
        # Simplified: game ends if one player has no pieces
        return np.all(state >= 0) or np.all(state <= 0)

    def get_reward(self, state):
        if np.all(state >= 0):
            return 1
        elif np.all(state <= 0):
            return -1
        else:
            return 0

    def get_action_size(self):
        return self.action_size

    def get_board_size(self):
        return self.board_size

    def set_complexity(self, complexity):
        self.complexity = complexity

    def get_canonical_form(self, state, player):
        return state * player

    def get_opponent(self, player):
        return -player

    def get_value_and_terminated(self, state, action):
        next_state = self.get_next_state(state, action)
        value = self.get_reward(next_state)
        terminated = self.is_terminal(next_state)
        return value, terminated

    def get_game_phase(self, state=None):
        if state is None:
            state = self.get_initial_state()

        # Count the number of pieces on the board
        piece_count = np.sum(np.abs(state))

        # Calculate phase probabilities
        opening_threshold = self.initial_pieces * 0.8
        middlegame_threshold = self.initial_pieces * 0.4

        if piece_count >= opening_threshold:
            opening_prob = 1.0
            middlegame_prob = 0.0
            endgame_prob = 0.0
        elif piece_count >= middlegame_threshold:
            opening_prob = (piece_count - middlegame_threshold) / (opening_threshold - middlegame_threshold)
            middlegame_prob = 1 - opening_prob
            endgame_prob = 0.0
        else:
            opening_prob = 0.0
            middlegame_prob = piece_count / middlegame_threshold
            endgame_prob = 1 - middlegame_prob

        return torch.tensor([opening_prob, middlegame_prob, endgame_prob])
