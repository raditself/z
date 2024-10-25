# Game Analysis Module

The Game Analysis module provides tools for analyzing games and evaluating positions.

## Class: GameAnalyzer

### Methods

#### __init__(self, game)
Initialize the GameAnalyzer.

- Parameters:
  - game: An instance of the Game class

#### analyze_game(self, game_history)
Analyze a complete game.

- Parameters:
  - game_history: A list of game states representing the game's progression
- Returns:
  - A dictionary containing analysis results, including move quality and critical positions

#### evaluate_position(self, state)
Evaluate a single game position.

- Parameters:
  - state: The game state to evaluate
- Returns:
  - A float representing the evaluation of the position (-1 to 1)

#### find_best_move(self, state)
Find the best move in a given position.

- Parameters:
  - state: The current game state
- Returns:
  - The best move and its evaluation

#### compare_moves(self, state, move1, move2)
Compare two moves in a given position.

- Parameters:
  - state: The current game state
  - move1: The first move to compare
  - move2: The second move to compare
- Returns:
  - A comparison result indicating which move is better

