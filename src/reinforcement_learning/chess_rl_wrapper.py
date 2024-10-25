
from typing import Tuple, List
import numpy as np
from src.games.chess.chess_game import ChessGame

class ChessRLWrapper:
    def __init__(self):
        self.game = ChessGame()
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
        piece_map = {'.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                     'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}
        return piece_map.get(piece, 0)

    def _generate_action_space(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        actions = []
        for from_row in range(8):
            for from_col in range(8):
                for to_row in range(8):
                    for to_col in range(8):
                        actions.append(((from_row, from_col), (to_row, to_col)))
        return actions
