
import os
import argparse
import torch
import torch.multiprocessing as mp
from distributed_alphazero import setup, cleanup, DistributedAlphaZero
from games.tictactoe import TicTacToe  # Assuming TicTacToe is implemented

def run(rank, world_size, args):
    setup(rank, world_size)
    game = TicTacToe()
    distributed_alphazero = DistributedAlphaZero(game, args, rank, world_size)
    distributed_alphazero.distributed_learn()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed AlphaZero')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of self-play episodes per iteration')
    parser.add_argument('--num_simulations', type=int, default=25, help='Number of MCTS simulations per move')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per training iteration')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for storing logs')
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"This script requires at least 2 GPUs to run, but only {num_gpus} GPU(s) are available.")
        exit(1)

    os.makedirs(args.log_dir, exist_ok=True)
    mp.spawn(run, args=(num_gpus, args), nprocs=num_gpus, join=True)
