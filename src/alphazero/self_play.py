


import numpy as np
import random
from .mcts import MCTS
from .game import Game
from .model import AlphaZeroNetwork
from .opening_book import OpeningBook
from .data_handler import DataHandler

class SelfPlay:
    def __init__(self, game: Game, model: AlphaZeroNetwork, num_simulations: int, data_dir: str, args):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.data_handler = DataHandler(data_dir)
        self.args = args
        self.mcts = MCTS(game, model, args)
        self.opening_book = OpeningBook()

    def play_game(self):
        state = self.game.get_initial_state()
        game_history = []
        move_count = 0

        while True:
            canonical_state = self.game.get_canonical_form(state)

            # Use opening book with some probability in the first few moves
            if move_count < 10 and random.random() < 0.5:
                chess_board = self.game.to_chess_board(state)
                book_move = self.opening_book.get_move(chess_board)
                if book_move:
                    action = self.game.chess_move_to_action(book_move)
                    action_probs = np.zeros(self.game.action_size())
                    action_probs[action] = 1
                else:
                    actions, action_probs = self.mcts.search(canonical_state)
            else:
                actions, action_probs = self.mcts.search(canonical_state)

            # Add exploration noise to the action probabilities
            action_probs = self.add_exploration_noise(action_probs)

            game_history.append((canonical_state, action_probs, state.to_play))

            # Choose action based on the action probabilities
            action = np.random.choice(len(action_probs), p=action_probs)
            state = self.game.get_next_state(state, action)
            move_count += 1

            if self.game.is_game_over(state):
                value = self.game.get_game_ended(state)
                return game_history, value

    def generate_data(self, num_games: int):
        examples = []

        for _ in range(num_games):
            game_history, value = self.play_game()
            examples.extend([(state, action_prob, value) for state, action_prob, _ in game_history])

        game_states = [self.game.get_encoded_state(state) for state, _, _ in examples]
        policy_targets = [action_prob for _, action_prob, _ in examples]
        value_targets = [value for _, _, value in examples]

        self.data_handler.save_game_data(game_states, policy_targets, value_targets)

    def add_exploration_noise(self, action_probs):
        noise = np.random.dirichlet([self.args.dirichlet_alpha] * len(action_probs))
        return (1 - self.args.exploration_fraction) * np.array(action_probs) + self.args.exploration_fraction * noise

# Keep the existing execute_self_play function and the rest of the file as is


def execute_self_play(model, game, num_games, args):
    data_dir = "./data"  # You may want to make this configurable
    self_play = SelfPlay(game, model, args.num_simulations, data_dir, args)
    self_play.generate_data(num_games)

if __name__ == "__main__":
    # Example usage
    from .checkers import Checkers
    game = Checkers()
    model = AlphaZeroNetwork(game)
    args = type('Args', (), {
        'num_simulations': 100,
        'dirichlet_alpha': 0.3,
        'exploration_fraction': 0.25
    })()
    execute_self_play(model, game, num_games=10, args=args)


