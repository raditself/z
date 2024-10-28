# GUI Module

The GUI module provides a graphical user interface for playing the Checkers game.

## Functions

### draw_board(game)
Draw the current state of the Checkers board.

- Parameters:
  - game: An instance of the CheckersGame class

### get_row_col_from_mouse(pos)
Convert mouse position to board coordinates.

- Parameters:
  - pos: A tuple containing the (x, y) position of the mouse
- Returns:
  - A tuple containing the (row, col) position on the board

### main()
The main game loop that handles user input and updates the display.

## Usage

To run the Checkers game with the GUI:

1. Ensure you have Pygame installed: `pip install pygame`
2. Run the script: `python src/games/checkers_gui.py`

## Controls

- Click on a piece to select it
- Click on a valid square to move the selected piece
- Close the window to exit the game

