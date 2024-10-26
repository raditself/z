
import argparse
import os
from .tictactoe import TicTacToe
from .distributed_alphazero import distributed_main

def main():
    parser = argparse.ArgumentParser(description='Train Distributed AlphaZero for Tic-Tac-Toe')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of self-play episodes per iteration')
    parser.add_argument('--num_simulations', type=int, default=25, help='Number of MCTS simulations per move')
    parser.add_argument('--c_puct', type=float, default=1.0, help='Exploration constant for PUCT')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per training iteration')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Interval for saving model checkpoints')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for distributed training')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for storing logs')
    parser.add_argument('--eval_games', type=int, default=100, help='Number of games to play during evaluation')

    args = parser.parse_args()

    # Create log directory if it doesn't exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    game = TicTacToe()
    distributed_main(game, args, args.num_workers)

if __name__ == '__main__':
    main()
