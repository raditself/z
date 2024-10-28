
import numpy as np
from .game import Game

class TicTacToe(Game):
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        col = action % self.column_count
        state[row, col] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def get_value_and_terminated(self, state, action):
        row = action // self.column_count
        col = action % self.column_count
        player = state[row, col]

        # Check row
        if np.all(state[row] == player):
            return 1, True
        # Check column
        if np.all(state[:, col] == player):
            return 1, True
        # Check diagonals
        if row == col and np.all(np.diag(state) == player):
            return 1, True
        if row + col == 2 and np.all(np.diag(np.fliplr(state)) == player):
            return 1, True

        # Check for draw
        if np.all(state != 0):
            return 0, True

        return 0, False

    def get_opponent(self, player):
        return -player

    def get_canonical_form(self, state, player):
        return state * player

    def get_action_size(self):
        return self.action_size

    def get_board_size(self):
        return (self.row_count, self.column_count)

    def __str__(self):
        return "TicTacToe"
