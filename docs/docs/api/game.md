# Game Module

The Game module defines the rules and logic for the games being played by the AI.

## Class: CheckersGame

### Methods

#### __init__(self)
Initialize the Checkers game.

#### get_initial_board(self)
Get the initial board setup for Checkers.

- Returns:
  - A 2D list representing the initial board state

#### get_valid_moves(self, player)
Get all valid moves for the given player.

- Parameters:
  - player: The player (1 for white, -1 for black)
- Returns:
  - A list of valid moves

#### get_piece_moves(self, row, col)
Get valid moves for a specific piece.

- Parameters:
  - row: The row of the piece
  - col: The column of the piece
- Returns:
  - A list of valid moves for the piece

#### make_move(self, move)
Make a move on the board.

- Parameters:
  - move: A tuple representing the move (start_row, start_col, end_row, end_col)

#### is_game_over(self)
Check if the game is over.

- Returns:
  - Boolean indicating whether the game is over

#### get_winner(self)
Get the winner of the game.

- Returns:
  - 1 for white win, -1 for black win, 0 for draw or game not over

