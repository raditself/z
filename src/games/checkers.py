import numpy as np

class CheckersGame:
    def __init__(self):
        self.board_size = 8
        self.action_size = self.board_size * self.board_size

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
