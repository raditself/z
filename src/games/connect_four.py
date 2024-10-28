
import numpy as np

class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.action_size = self.columns
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0

    def get_initial_state(self):
        return self.board

    def get_next_state(self, state, action):
        new_state = state.copy()
        for row in range(self.rows - 1, -1, -1):
            if new_state[row, action] == 0:
                new_state[row, action] = self.current_player
                break
        return new_state

    def get_valid_actions(self, state):
        return (state[0] == 0).astype(np.int8)

    def is_terminal(self, state):
        # Check for a win
        for player in [1, -1]:
            player_state = (state == player)
            # Check horizontal locations
            for c in range(self.columns - 3):
                for r in range(self.rows):
                    if np.all(player_state[r, c:c+4]):
                        return True
            # Check vertical locations
            for c in range(self.columns):
                for r in range(self.rows - 3):
                    if np.all(player_state[r:r+4, c]):
                        return True
            # Check positively sloped diagonals
            for c in range(self.columns - 3):
                for r in range(self.rows - 3):
                    if np.all(player_state[range(r, r+4), range(c, c+4)]):
                        return True
            # Check negatively sloped diagonals
            for c in range(self.columns - 3):
                for r in range(3, self.rows):
                    if np.all(player_state[range(r, r-4, -1), range(c, c+4)]):
                        return True
        # Check if board is full
        return np.all(state != 0)

    def get_reward(self, state):
        if not self.is_terminal(state):
            return 0
        for player in [1, -1]:
            player_state = (state == player)
            # Check horizontal locations
            for c in range(self.columns - 3):
                for r in range(self.rows):
                    if np.all(player_state[r, c:c+4]):
                        return player
            # Check vertical locations
            for c in range(self.columns):
                for r in range(self.rows - 3):
                    if np.all(player_state[r:r+4, c]):
                        return player
            # Check positively sloped diagonals
            for c in range(self.columns - 3):
                for r in range(self.rows - 3):
                    if np.all(player_state[range(r, r+4), range(c, c+4)]):
                        return player
            # Check negatively sloped diagonals
            for c in range(self.columns - 3):
                for r in range(3, self.rows):
                    if np.all(player_state[range(r, r-4, -1), range(c, c+4)]):
                        return player
        return 0  # Draw

    def get_canonical_state(self, state, player):
        return state * player

    def get_symmetries(self, state, pi):
        return [(state, pi), (np.fliplr(state), np.flipud(pi))]

    def string_representation(self, state):
        return state.tostring()

if __name__ == "__main__":
    connect_four = ConnectFour()
    print("Board shape:", connect_four.board.shape)
    print("Action size:", connect_four.action_size)
    print("Initial state:\n", connect_four.get_initial_state())
