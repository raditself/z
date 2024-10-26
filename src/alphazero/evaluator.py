
import torch
import numpy as np
from .mcts import MCTS

class AlphaZeroEvaluator:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self, model1, model2, num_games=100):
        wins = {1: 0, -1: 0, 0: 0}  # 1: model1 wins, -1: model2 wins, 0: draw
        
        for game_num in range(num_games):
            state = self.game.get_initial_state()
            player = 1
            mcts1 = MCTS(self.game, model1, self.args)
            mcts2 = MCTS(self.game, model2, self.args)

            while True:
                if player == 1:
                    action = self.get_action(state, mcts1)
                else:
                    action = self.get_action(state, mcts2)

                state = self.game.get_next_state(state, action, player)
                value, is_terminal = self.game.get_value_and_terminated(state, action)

                if is_terminal:
                    wins[value * player] += 1
                    break

                player = self.game.get_opponent(player)

            if game_num % 10 == 0:
                print(f"Completed {game_num + 1} evaluation games")

        return wins

    def get_action(self, state, mcts):
        for _ in range(self.args.num_simulations):
            mcts.search(state)

        pi = mcts.get_action_prob(state)
        action = np.argmax(pi)
        return action

def load_model(model_class, model_path, game, args):
    model = model_class(game.board_size, game.action_size)
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model

