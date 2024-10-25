
# Chess and Checkers AI Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage Examples](#usage-examples)
4. [API Reference](#api-reference)
   - [Chess AI](#chess-ai-api-reference)
   - [Checkers](#checkers-api-reference)

## Project Overview

This project implements AI players for both Chess and Checkers games. It uses various AI techniques including minimax algorithm with alpha-beta pruning for Chess, and a custom implementation for Checkers. The project aims to provide a flexible and extensible framework for board game AI development and analysis.

Key features:
- Chess AI using minimax algorithm with alpha-beta pruning
- Checkers game implementation with AI support
- Reinforcement Learning capabilities for both Chess and Checkers
- Distributed training system for faster AI improvement
- Support for chess variants (Chess960 and King of the Hill)
- Tournament system for AI competition
- User-friendly GUI for playing against the AI
- Opening book implementation for improved early game play
- Support for time controls in game implementation

## Installation

To install the Chess and Checkers AI project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/raditself/z.git
   cd z
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Set up the distributed training system if you plan to use multiple machines for training.

## Usage Examples

### Playing Chess against the AI

```python
from src.alphazero.chess_ai import ChessAI
import chess

# Create a new chess board
board = chess.Board()

# Initialize the Chess AI with a search depth of 3
ai = ChessAI(depth=3)

while not board.is_game_over():
    if board.turn == chess.WHITE:
        # Human player's turn (White)
        move = input("Enter your move (e.g., 'e2e4'): ")
        board.push_san(move)
    else:
        # AI's turn (Black)
        move = ai.get_best_move(board)
        board.push(move)
    
    print(board)
    print()

print("Game Over")
print("Result:", board.result())
```

### Playing Checkers

```python
from src.games.checkers import Checkers

# Initialize a new Checkers game
game = Checkers()

while not game.is_game_over():
    print(game)
    print(f"Current player: {game.get_current_player()}")
    
    valid_moves = game.get_valid_moves()
    print("Valid moves:", valid_moves)
    
    # For simplicity, we'll just choose the first valid move
    move = valid_moves[0]
    game.make_move(move)

print("Game Over")
winner = game.get_winner()
print(f"Winner: Player {winner}")
```

## API Reference

The API Reference section includes detailed documentation for the main classes in the project. Please refer to the individual sections for Chess AI and Checkers for more information.


# Chess AI API Reference

## Class: ChessAI

The `ChessAI` class implements a chess AI using the minimax algorithm with alpha-beta pruning.

### Constructor

#### `__init__(self, depth=3)`

Initializes a new ChessAI instance.

- Parameters:
  - `depth` (int, optional): The depth of the minimax search tree. Default is 3.

### Methods

#### `get_best_move(self, board)`

Finds the best move for the current board position.

- Parameters:
  - `board` (chess.Board): The current chess board state.
- Returns:
  - chess.Move: The best move found by the AI.

#### `minimax(self, board, depth, alpha, beta, maximizing_player)`

Implements the minimax algorithm with alpha-beta pruning.

- Parameters:
  - `board` (chess.Board): The current chess board state.
  - `depth` (int): The current depth in the search tree.
  - `alpha` (float): The alpha value for alpha-beta pruning.
  - `beta` (float): The beta value for alpha-beta pruning.
  - `maximizing_player` (bool): True if the current player is maximizing, False otherwise.
- Returns:
  - float: The evaluation score of the board position.

#### `evaluate_board(self, board)`

Evaluates the current board position.

- Parameters:
  - `board` (chess.Board): The chess board to evaluate.
- Returns:
  - float: The evaluation score of the board position.

### Implementation Details

- The AI uses a simple material-based evaluation function.
- It implements alpha-beta pruning to optimize the minimax algorithm.
- The evaluation function assigns the following values to pieces:
  - Pawn: 100
  - Knight: 320
  - Bishop: 330
  - Rook: 500
  - Queen: 900
  - King: 20000
- Checkmate is valued at ±10000 depending on the side.
- Stalemate and insufficient material are valued at 0.

# Checkers AI API Reference

## Class: Checkers

The `Checkers` class implements the game logic for a standard game of Checkers.

### Constructor

#### `__init__(self)`

Initializes a new Checkers game instance.

### Methods

#### `initialize_board(self)`

Sets up the initial board state for a new game of Checkers.

#### `get_valid_moves(self)`

Returns a list of all valid moves for the current player.

- Returns:
  - list: A list of valid moves, where each move is represented as a tuple (start_row, start_col, end_row, end_col).

#### `get_piece_moves(self, row, col)`

Returns a list of valid moves for a specific piece.

- Parameters:
  - `row` (int): The row of the piece.
  - `col` (int): The column of the piece.
- Returns:
  - list: A list of valid moves for the specified piece.

#### `make_move(self, move)`

Executes a move on the board.

- Parameters:
  - `move` (tuple): A tuple representing the move (start_row, start_col, end_row, end_col).

#### `is_game_over(self)`

Checks if the game has ended.

- Returns:
  - bool: True if the game is over, False otherwise.

#### `get_winner(self)`

Determines the winner of the game.

- Returns:
  - int or None: The player number (1 or 2) of the winner, or None if the game is not over.

#### `get_state(self)`

Returns the current state of the board.

- Returns:
  - numpy.ndarray: A copy of the current board state.

#### `get_current_player(self)`

Returns the current player's number.

- Returns:
  - int: The current player (1 or 2).

#### `undo_move(self, move)`

Reverses a move on the board.

- Parameters:
  - `move` (tuple): A tuple representing the move to undo (start_row, start_col, end_row, end_col).

#### `__str__(self)`

Returns a string representation of the current board state.

- Returns:
  - str: A string representation of the board, where '.' represents an empty square, 'x' represents player 1's pieces, and 'o' represents player 2's pieces.

### Implementation Details

- The board is represented as an 8x8 numpy array.
- Player 1's pieces are represented by 1, Player 2's pieces by 2, and empty squares by 0.
- The game uses a move cache to optimize performance when calculating valid moves.
- The game supports standard Checkers rules, including capturing moves.
