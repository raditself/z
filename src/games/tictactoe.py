
import numpy as np
from src.alphazero.abstract_game import AbstractGame

class TicTacToeGame(AbstractGame):
    def __init__(self):
        self.board_size = 3
        self._action_size = self.board_size * self.board_size

    def get_initial_state(self):
        return np.zeros((self.board_size, self.board_size))

    def get_next_state(self, state, action):
        player = 1 if np.sum(state) == 0 else -1
        next_state = state.copy()
        next_state[action // self.board_size, action % self.board_size] = player
        return next_state

    def get_valid_moves(self, state):
        return (state == 0).flatten()

    def is_game_over(self, state):
        return self.get_winner(state) != 0 or np.all(state != 0)

    def get_winner(self, state):
        for player in [1, -1]:
            # Check rows, columns and diagonals
            if np.any(np.all(state == player, axis=0)) or                np.any(np.all(state == player, axis=1)) or                np.all(np.diag(state) == player) or                np.all(np.diag(np.fliplr(state)) == player):
                return player
        return 0

    def get_canonical_state(self, state, player):
        return state * player

    def state_to_tensor(self, state):
        return np.array([state == 1, state == -1, state == 0]).astype(np.float32)

    def action_to_move(self, action):
        return (action // self.board_size, action % self.board_size)

    def move_to_action(self, move):
        return move[0] * self.board_size + move[1]

    @property
    def action_size(self):
        return self._action_size

    @property
    def state_shape(self):
        return (3, self.board_size, self.board_size)

    def render(self, state):
        print(state)

    def get_current_player(self, state):
        return 1 if np.sum(state) == 0 else -1
