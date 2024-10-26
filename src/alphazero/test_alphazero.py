
import sys
import os
import numpy as np

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.alphazero.alphazero import AlphaZero
from src.games.tictactoe import TicTacToeGame
from src.games.checkers_v2 import CheckersGame
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test AlphaZero with different games')
    parser.add_argument('--game', type=str, choices=['tictactoe', 'checkers'], required=True, help='Game to play')
    args = parser.parse_args()

    if args.game == 'tictactoe':
        game = TicTacToeGame()
    elif args.game == 'checkers':
        game = CheckersGame()

    # Set up AlphaZero arguments
    class AlphaZeroArgs:
        def __init__(self):
            self.num_simulations = 50
            self.c_puct = 1.0
            self.c_pw = 1.0
            self.alpha_pw = 0.5
            self.lr = 0.001
            self.weight_decay = 0.0001
            self.epochs = 10
            self.batch_size = 64
            self.num_iterations = 10
            self.num_episodes = 10
            self.checkpoint_interval = 5

    alphazero_args = AlphaZeroArgs()

    # Initialize AlphaZero
    alphazero = AlphaZero(game, alphazero_args)

    # Run a short learning process
    alphazero.learn()

    # Play a game against a random player
    state = game.get_initial_state()
    game.render(state)

    while not game.is_game_over(state):
        if game.get_current_player(state) == 1:
            action = alphazero.play(state)
        else:
            valid_moves = game.get_valid_moves(state)
            action = np.random.choice(len(valid_moves), p=valid_moves/np.sum(valid_moves))

        state = game.get_next_state(state, action)
        game.render(state)

    winner = game.get_winner(state)
    if winner == 1:
        print("AlphaZero wins!")
    elif winner == -1:
        print("Random player wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()
