
# AlphaZero Implementation

This project is an implementation of the AlphaZero algorithm for chess. It includes a neural network model, Monte Carlo Tree Search (MCTS), and self-play functionality.

## Features

- Chess game implementation
- Neural network model for policy and value prediction
- Monte Carlo Tree Search (MCTS) for action selection
- Self-play functionality for training
- Model export and import capabilities
- Web interface for interacting with the trained model

## Running Tests

To run all tests for the AlphaZero implementation, use the following command from the project root directory:

```
python run_tests.py
```

This will execute all unit tests and integration tests, providing a summary of the results.

## Project Structure

- `src/alphazero/`: Contains the core implementation of the AlphaZero algorithm
- `src/web/`: Contains the web interface for interacting with the trained model
- `tests/`: Contains unit tests and integration tests
- `run_tests.py`: Script to run all tests

## Getting Started

1. Clone the repository
2. Install the required dependencies (list them here or refer to a requirements.txt file)
3. Run the tests to ensure everything is set up correctly
4. Start training the model or use a pre-trained model

For more detailed instructions, please refer to the documentation.

## Project Status and Roadmap

1. Implement a distributed training system to leverage multiple machines for faster training.
   Status: Implemented in train.py using PyTorch's DistributedDataParallel.

2. Add support for more chess variants or even different board games.
   Status: Partially implemented. The Game class supports different variants, and we've added support for Chess960 and King of the Hill variants in chess_variants.py. Full implementation for other board games is not complete.

3. Implement a tournament system where different versions of the AI can compete against each other.
   Status: Implemented in tournament.py.

4. Optimize the neural network architecture for better performance, possibly using techniques like neural architecture search.
   Status: Implemented. We've added neural architecture search in neural_architecture_search.py and integrated it into the training process in train.py.

5. Implement a user-friendly GUI for playing against the trained AI.
   Status: Implemented in chess_gui.py using Pygame.

6. Add support for loading and analyzing real-world chess games to improve the AI's opening knowledge.
   Status: Implemented in game_analysis.py and integrated into chess_gui.py.

7. Implement an opening book to improve the AI's play in the early game.
   Status: Implemented.

8. Add support for time controls in the game implementation.
   Status: Implemented in chess_logic.py and integrated into chess_gui.py.

9. Implement a system for continuous integration and deployment (CI/CD) to automatically run tests and deploy updates.
   Status: Implemented. GitHub Actions workflow created in .github/workflows/python-ci.yml to run tests and linting on pushes and pull requests to the master branch.

10. Create comprehensive documentation for the project, including API references and usage examples.
    Status: Partially implemented through code comments. Full documentation not yet created.

### Remaining tasks to focus on:
1. Complete the implementation of different board games beyond chess variants.
2. Enhance the GUI to support all implemented chess variants and potentially other board games.
3. Create comprehensive documentation for the project, including setup instructions, API references, and usage examples.
4. Conduct thorough testing of all implemented features and optimize performance where necessary.

