
from abc import ABC, abstractmethod
import numpy as np

class AbstractGame(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_initial_state(self):
        pass

    @abstractmethod
    def get_next_state(self, state, action):
        pass

    @abstractmethod
    def get_valid_moves(self, state):
        pass

    @abstractmethod
    def is_game_over(self, state):
        pass

    @abstractmethod
    def get_winner(self, state):
        pass

    @abstractmethod
    def get_canonical_state(self, state, player):
        pass

    @abstractmethod
    def state_to_tensor(self, state):
        pass

    @abstractmethod
    def action_to_move(self, action):
        pass

    @abstractmethod
    def move_to_action(self, move):
        pass

    @property
    @abstractmethod
    def action_size(self):
        pass

    @property
    @abstractmethod
    def state_shape(self):
        pass

    @abstractmethod
    def render(self, state):
        pass

    @abstractmethod
    def get_current_player(self, state):
        pass
