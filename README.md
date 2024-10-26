
# Distributed AlphaZero Implementation

This project implements a distributed version of the AlphaZero algorithm for the game of Tic-Tac-Toe. It uses PyTorch for neural network training and implements distributed training using PyTorch's DistributedDataParallel.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy

## Project Structure

- `src/alphazero/`
  - `alphazero_net.py`: Defines the neural network architecture
  - `mcts.py`: Implements the Monte Carlo Tree Search algorithm
  - `alphazero.py`: Contains the core AlphaZero algorithm
  - `distributed_alphazero.py`: Implements the distributed version of AlphaZero
  - `evaluator.py`: Provides functionality to evaluate trained models
  - `game.py`: Defines the abstract base class for games
  - `tictactoe.py`: Implements the Tic-Tac-Toe game
  - `logger.py`: Handles logging and visualization of training progress
  - `run_distributed_training.py`: Main script to run the distributed training

## How to Run

To start the distributed training process, run the following command:

```
python -m src.alphazero.run_distributed_training
```

You can customize the training process by passing additional arguments. For example:

```
python -m src.alphazero.run_distributed_training --num_iterations 200 --num_workers 8 --eval_games 200
```

## Command-line Arguments

- `--num_iterations`: Number of training iterations (default: 100)
- `--num_episodes`: Number of self-play episodes per iteration (default: 100)
- `--num_simulations`: Number of MCTS simulations per move (default: 25)
- `--c_puct`: Exploration constant for PUCT (default: 1.0)
- `--lr`: Learning rate for the optimizer (default: 0.001)
- `--weight_decay`: Weight decay for the optimizer (default: 1e-4)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of epochs per training iteration (default: 10)
- `--checkpoint_interval`: Interval for saving model checkpoints (default: 10)
- `--num_workers`: Number of worker processes for distributed training (default: 4)
- `--log_dir`: Directory for storing logs (default: 'logs')
- `--eval_games`: Number of games to play during evaluation (default: 100)

## Monitoring Training Progress

Training progress can be monitored using TensorBoard. To start TensorBoard, run:

```
tensorboard --logdir=logs
```

Then open a web browser and go to `http://localhost:6006` to view the training metrics.

## Extending to Other Games

To extend this implementation to other games, create a new game class that inherits from the `Game` base class in `game.py`. Implement all the required methods, and then update the `run_distributed_training.py` script to use your new game class.

