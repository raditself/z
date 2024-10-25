# AlphaZero-like Board Game AI

This project implements an AlphaZero-like AI system for playing board games, currently supporting Chess and Checkers.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Contributing](#contributing)
7. [License](#license)

## Overview

This project aims to create a versatile AI system capable of playing various board games using techniques inspired by AlphaZero. The current implementation supports Chess and Checkers, with the potential to add more games in the future.

## Features

- Support for multiple board games (Chess and Checkers)
- Graphical user interface for playing against the AI
- AI opponent using Monte Carlo Tree Search (MCTS) and neural networks
- Game analysis tools for chess games
- Ability to load and analyze PGN files for chess games

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/alphazero-board-games.git
   cd alphazero-board-games
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To start the game GUI:

```
python src/alphazero/game_gui.py
```

By default, the Chess game will be loaded. To play Checkers, modify the last line of the game_gui.py file to:

```python
game_gui = GameGUI(game_type='checkers')
```

## Project Structure

- `src/`: Contains the main source code
  - `alphazero/`: AlphaZero-related implementations
    - `chess_logic.py`: Chess game logic
    - `checkers.py`: Checkers game logic
    - `game_gui.py`: Graphical user interface for games
    - `chess_ai.py`: Chess AI implementation
    - `game_analysis.py`: Game analysis tools
  - `games/`: Game-specific implementations
- `tests/`: Unit tests
- `static/`: Static files (e.g., images)
- `templates/`: HTML templates (if applicable)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
