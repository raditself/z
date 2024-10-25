
from typing import Tuple, List
import numpy as np
from src.games.checkers.checkers_game import CheckersGame

class CheckersRLWrapper:
    def __init__(self):
        self.game = CheckersGame()
        self.action_space = self._generate_action_space()

    def reset(self) -> List[List[int]]:
        self.game.reset()
        return self._board_to_state()

    def step(self, action: int) -> Tuple[List[List[int]], float, bool, dict]:
        move = self.action_space[action]
        is_valid = self.game.make_move(move[0], move[1])
        
        if not is_valid:
            return self._board_to_state(), -10, True, {}

        reward = 0
        done = False

        if self.game.is_game_over():
            done = True
            if self.game.get_winner() == self.game.current_player:
                reward = 1
            elif self.game.get_winner() is None:
                reward = 0
            else:
                reward = -1

        return self._board_to_state(), reward, done, {}

    def _board_to_state(self) -> List[List[int]]:
        return [[self._piece_to_int(piece) for piece in row] for row in self.game.board]

    def _piece_to_int(self, piece: str) -> int:
        piece_map = {'.': 0, 'r': 1, 'R': 2, 'b': -1, 'B': -2}
        return piece_map.get(piece, 0)

    def _generate_action_space(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        actions = []
        for from_row in range(8):
            for from_col in range(8):
                for to_row in range(8):
                    for to_col in range(8):
                        actions.append(((from_row, from_col), (to_row, to_col)))
        return actions
