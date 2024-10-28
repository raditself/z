
import pytest
import numpy as np
from src.games.chess import Chess
from src.alphazero.mcts import AdaptiveMCTS
from src.alphazero.neural_network import DynamicNeuralNetwork
from train_alphazero import train_alphazero

@pytest.fixture
def game():
    return Chess()

@pytest.fixture
def network(game):
    return DynamicNeuralNetwork(game)

@pytest.fixture
def mcts(game, network):
    return AdaptiveMCTS(game, network)

def test_chess_game(game):
    assert game.board_size == 8
    assert game.action_size == 64 * 64
    initial_state = game.get_initial_state()
    assert initial_state.shape == (8, 8, 12)

def test_neural_network(game, network):
    state = game.get_initial_state()
    policy, value = network.predict(state)
    assert policy.shape == (game.action_size,)
    assert isinstance(value, float)

def test_mcts(game, mcts):
    state = game.get_initial_state()
    action = mcts.search(state)
    assert 0 <= action < game.action_size

def test_train_alphazero(game):
    trained_network = train_alphazero(game, num_iterations=2, num_episodes=2, num_epochs=2)
    assert isinstance(trained_network, DynamicNeuralNetwork)

if __name__ == "__main__":
    pytest.main([__file__])
