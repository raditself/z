
from abc import ABC, abstractmethod
import numpy as np

class Game(ABC):
    @abstractmethod
    def get_initial_state(self):
        pass

    @abstractmethod
    def get_next_state(self, state, action, player):
        pass

    @abstractmethod
    def get_valid_moves(self, state):
        pass

    @abstractmethod
    def get_value_and_terminated(self, state, action):
        pass

    @abstractmethod
    def get_opponent(self, player):
        pass

    @abstractmethod
    def get_canonical_form(self, state, player):
        pass

    @abstractmethod
    def get_action_size(self):
        pass

    @abstractmethod
    def get_board_size(self):
        pass

    @abstractmethod
    def set_complexity(self, complexity):
        pass

    def __str__(self):
        return self.__class__.__name__
