
import numpy as np
from src.alphazero.mcts import MCTS

class SelfPlay:
    def __init__(self, game, model, num_mcts_sims):
        self.game = game
        self.model = model
        self.mcts = MCTS(game, model, num_mcts_sims)

    def execute_episode(self):
        train_examples = []
        state = self.game.get_initial_state()
        self.game.reset()
        step = 0

        while True:
            step += 1
            temp = int(step < 30)  # temperature threshold set to 30 moves

            pi = self.mcts.search(state)
            train_examples.append([state, pi, None])

            action = np.random.choice(len(pi), p=pi)
            state = self.game.get_next_state(action)

            if self.game.is_game_over():
                return [(x[0], x[1], self.game.get_winner() * ((-1) ** (x[0] != state))) for x in train_examples]

def execute_self_play(game, model, num_mcts_sims):
    self_play = SelfPlay(game, model, num_mcts_sims)
    return self_play.execute_episode()
