
import pytest
import torch
from src.alphazero.neural_network import DynamicNeuralNetwork
from src.games.chess import Chess

@pytest.fixture
def game():
    return Chess()

@pytest.fixture
def network(game):
    return DynamicNeuralNetwork(game)

def test_network_initialization(network):
    assert isinstance(network, DynamicNeuralNetwork)
    assert network.num_actions == 64 * 64  # for chess

def test_network_forward_pass(network, game):
    state = game.get_initial_state()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    policy, value = network(state_tensor)
    assert policy.shape == (1, 64 * 64)
    assert value.shape == (1, 1)

def test_network_predict(network, game):
    state = game.get_initial_state()
    policy, value = network.predict(state)
    assert policy.shape == (64 * 64,)
    assert isinstance(value, float)
