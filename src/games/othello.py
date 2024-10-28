
import numpy as np

class Othello:
    def __init__(self):
        self.board_size = 8
        self.action_size = self.board_size ** 2
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.board[3, 3] = self.board[4, 4] = 1
        self.board[3, 4] = self.board[4, 3] = -1
        self.current_player = 1
        self.move_count = 0

    def get_initial_state(self):
        return self.board

    def get_next_state(self, state, action):
        x, y = action // self.board_size, action % self.board_size
        new_state = state.copy()
        new_state[x, y] = self.current_player
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            self._flip_direction(new_state, x, y, dx, dy)
        return new_state

    def _flip_direction(self, state, x, y, dx, dy):
        if not self._is_valid_direction(state, x, y, dx, dy):
            return
        cur_x, cur_y = x + dx, y + dy
        while 0 <= cur_x < self.board_size and 0 <= cur_y < self.board_size:
            if state[cur_x, cur_y] == self.current_player:
                while (cur_x, cur_y) != (x, y):
                    cur_x, cur_y = cur_x - dx, cur_y - dy
                    state[cur_x, cur_y] = self.current_player
                return
            cur_x, cur_y = cur_x + dx, cur_y + dy

    def _is_valid_direction(self, state, x, y, dx, dy):
        cur_x, cur_y = x + dx, y + dy
        if not (0 <= cur_x < self.board_size and 0 <= cur_y < self.board_size):
            return False
        if state[cur_x, cur_y] != -self.current_player:
            return False
        while 0 <= cur_x < self.board_size and 0 <= cur_y < self.board_size:
            if state[cur_x, cur_y] == 0:
                return False
            if state[cur_x, cur_y] == self.current_player:
                return True
            cur_x, cur_y = cur_x + dx, cur_y + dy
        return False

    def get_valid_actions(self, state):
        valid = np.zeros(self.action_size, dtype=np.int8)
        for action in range(self.action_size):
            x, y = action // self.board_size, action % self.board_size
            if state[x, y] == 0 and self._is_valid_move(state, x, y):
                valid[action] = 1
        return valid

    def _is_valid_move(self, state, x, y):
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            if self._is_valid_direction(state, x, y, dx, dy):
                return True
        return False

    def is_terminal(self, state):
        return np.all(self.get_valid_actions(state) == 0)

    def get_reward(self, state):
        if not self.is_terminal(state):
            return 0
        player_score = np.sum(state == 1)
        opponent_score = np.sum(state == -1)
        if player_score > opponent_score:
            return 1
        elif player_score < opponent_score:
            return -1
        else:
            return 0

    def get_canonical_state(self, state, player):
        return state * player

    def get_symmetries(self, state, pi):
        symmetries = []
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    new_state = np.rot90(state, i)
                    if j == 1:
                        new_state = np.fliplr(new_state)
                    if k == 1:
                        new_state = np.flipud(new_state)
                    new_pi = np.rot90(pi.reshape(self.board_size, self.board_size), i)
                    if j == 1:
                        new_pi = np.fliplr(new_pi)
                    if k == 1:
                        new_pi = np.flipud(new_pi)
                    symmetries.append((new_state, new_pi.flatten()))
        return symmetries

    def string_representation(self, state):
        return state.tostring()

if __name__ == "__main__":
    othello = Othello()
    print("Board size:", othello.board_size)
    print("Action size:", othello.action_size)
    print("Initial state:\n", othello.get_initial_state())
