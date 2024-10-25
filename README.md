
# AI Board Games

This project implements AI players for Chess and Checkers using various techniques, including Reinforcement Learning.

## Project Structure

```
.
├── docs/
│   └── ...
├── src/
│   ├── alphazero/
│   │   └── ...
│   ├── games/
│   │   └── checkers.py
│   ├── reinforcement_learning/
│   │   ├── rl_agent.py
│   │   ├── chess_rl_wrapper.py
│   │   ├── checkers_rl_wrapper.py
│   │   └── train_rl_agents.py
│   └── web/
│       └── ...
├── static/
│   └── ...
├── templates/
│   └── ...
├── tests/
│   ├── test_game_logic.py
│   └── test_rl_agent.py
├── unit/
│   └── ...
├── README.md
├── requirements.txt
└── ...
```

## Reinforcement Learning

The project now includes a Reinforcement Learning implementation for both Chess and Checkers. The RL agents use Q-learning to improve their gameplay through self-play.

To train and test the RL agents, run:

```
python src/reinforcement_learning/train_rl_agents.py
```

This script will train RL agents for both Chess and Checkers, and then test their performance.

## Running Tests

To run the tests for the Reinforcement Learning implementation, use:

```
python -m pytest tests/test_rl_agent.py
```

## ...

