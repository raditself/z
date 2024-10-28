# AI Module

The AI module contains the core AlphaZero AI implementation.

## Class: AlphaZeroAI

### Methods

#### __init__(self, game, model)
Initialize the AlphaZero AI.

- Parameters:
  - game: An instance of the Game class
  - model: An instance of the neural network model

#### get_action(self, state)
Get the best action for the given state.

- Parameters:
  - state: The current game state
- Returns:
  - The chosen action

#### train(self, num_iterations)
Train the AI for a specified number of iterations.

- Parameters:
  - num_iterations: Number of training iterations

#### self_play(self)
Perform self-play to generate training data.

- Returns:
  - A list of (state, action_probabilities, value) tuples

#### get_action_probabilities(self, state)
Get the probability distribution over actions for a given state.

- Parameters:
  - state: The current game state
- Returns:
  - A list of probabilities for each possible action

