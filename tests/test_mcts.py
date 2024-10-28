
import pytest
from src.alphazero.mcts import AdaptiveMCTS
from src.games.chess import Chess

@pytest.fixture
def game():
    return Chess()

@pytest.fixture
def mcts(game):
    return AdaptiveMCTS(game, None)

def test_mcts_initialization(mcts):
    assert mcts.game is not None
    assert mcts.num_simulations == 800

def test_mcts_search(mcts, game):
    state = game.get_initial_state()
    action = mcts.search(state)
    assert 0 <= action < game.action_size

def test_mcts_update_hyperparameters(mcts):
    mcts.update_hyperparameters('opening')
    assert mcts.c_puct == 1.5
    mcts.update_hyperparameters('midgame')
    assert mcts.c_puct == 1.0
    mcts.update_hyperparameters('endgame')
    assert mcts.c_puct == 0.5
