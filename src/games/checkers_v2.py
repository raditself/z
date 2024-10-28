
import numpy as np
from src.alphazero.abstract_game import AbstractGame

class CheckersGame(AbstractGame):
    def __init__(self):
        self.board_size = 8
        self._action_size = self.board_size * self.board_size

    def get_initial_state(self):
        # Simplified initial state: 1 for player 1, -1 for player 2, 0 for empty
        board = np.zeros((self.board_size, self.board_size))
        board[:3, ::2] = 1
        board[-3:, 1::2] = -1
        return board

    def get_next_state(self, state, action):
        # Simplified: just flip a random piece
        next_state = state.copy()
        x, y = action // self.board_size, action % self.board_size
        next_state[x, y] *= -1
        return next_state

    def get_valid_moves(self, state):
        return (state == 0).flatten()

    def is_game_over(self, state):
        # Simplified: game ends if one player has no pieces
        return np.all(state >= 0) or np.all(state <= 0)

    def get_winner(self, state):
        if np.all(state >= 0):
            return 1
        elif np.all(state <= 0):
            return -1
        else:
            return 0

    def get_canonical_state(self, state, player):
        return state * player

    def state_to_tensor(self, state):
        # Convert the state to a 3D tensor (channels, height, width)
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
        # In Checkers, the player is determined by the number of moves made
        # Assuming player 1 starts, and players alternate turns
        return 1 if np.sum(np.abs(state)) % 2 == 0 else -1
