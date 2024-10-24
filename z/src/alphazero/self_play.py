
from .mcts import MCTS
from .game import Game
from .model import AlphaZeroNetwork
import numpy as np

def execute_self_play(model, game, num_games, args):
    mcts = MCTS(game, model, args)
    examples = []

    for _ in range(num_games):
        state = game.get_initial_state()
        game_history = []

        while True:
            canonical_state = game.get_canonical_form(state)
            actions, action_probs = mcts.search(canonical_state)

            # Add exploration noise to the action probabilities
            action_probs = add_exploration_noise(action_probs, args.dirichlet_alpha, args.exploration_fraction)

            game_history.append((canonical_state, action_probs, state.to_play))

            # Choose action based on the action probabilities
            action = np.random.choice(actions, p=action_probs)
            state = game.get_next_state(state, action)

            if game.is_game_over(state):
                value = game.get_game_ended(state)
                examples.extend([(state, action_prob, value) for state, action_prob, _ in game_history])
                break

    return examples

def add_exploration_noise(action_probs, alpha, exploration_fraction):
    noise = np.random.dirichlet([alpha] * len(action_probs))
    return (1 - exploration_fraction) * np.array(action_probs) + exploration_fraction * noise

