
# AlphaZero Chess and Checkers AI

This project implements an AI system for playing Chess and Checkers using the AlphaZero algorithm. It includes features such as neural network-based move evaluation, Monte Carlo Tree Search (MCTS), opening book support, game analysis, and a graphical user interface.

## Features

- AlphaZero-style AI for Chess and Checkers
- Neural Architecture Search for optimizing model architecture
- Tournament system for comparing different AI versions
- Opening book support for improved early game play
- Time control support for realistic gameplay
- Game analysis tools for reviewing and learning from games
- User-friendly GUI for playing against the AI

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/alphazero-chess-checkers.git
   cd alphazero-chess-checkers
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the AI

To train the AI for Chess:
```
python src/alphazero/train.py --game chess --iterations 100 --episodes 100 --epochs 10
```

To train the AI for Checkers:
```
python src/alphazero/train.py --game checkers --iterations 100 --episodes 100 --epochs 10
```

### Playing against the AI

To play Chess against the AI using the GUI:
```
python src/alphazero/game_gui.py --game chess --model_path path/to/chess_model.h5
```

To play Checkers against the AI using the GUI:
```
python src/alphazero/game_gui.py --game checkers --model_path path/to/checkers_model.h5
```

### Running a Tournament

To run a tournament between different AI versions:
```
python src/alphazero/tournament.py --game chess --model_paths model1.h5 model2.h5 model3.h5
```

### Analyzing Games

To analyze a chess game from a PGN file:
```
python src/alphazero/game_analysis.py --pgn_file game.pgn --model_path chess_model.h5
```


# AlphaZero Chess and Checkers AI

[... Keep the existing content ...]

## Running Tests

To run the entire test suite:

```
python -m unittest discover tests
```

To run specific test files:

```
python -m unittest tests/test_mcts.py
python -m unittest tests/test_chess_game.py
python -m unittest tests/test_model.py
python -m unittest tests/test_game_analysis.py
```

[... Keep the rest of the existing content ...]

## Project Structure

- `src/alphazero/`: Contains the core AlphaZero implementation
  - `mcts_nn.py`: Monte Carlo Tree Search with neural network guidance
  - `model.py`: Neural network model definitions
  - `train.py`: Training script for the AI
  - `chess_ai.py` and `checkers_ai.py`: Game-specific AI implementations
  - `tournament.py`: Tournament system for comparing AI versions
  - `game_gui.py`: Graphical user interface for playing against the AI
  - `game_analysis.py`: Tools for analyzing chess games
  - `opening_book.py`: Opening book implementation
  - `game_timer.py`: Time control implementation
- `src/games/`: Contains game logic for Chess and Checkers
- `tests/`: Unit tests and integration tests
- `.github/workflows/`: CI/CD configuration using GitHub Actions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
