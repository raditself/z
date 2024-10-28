
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reinforcement_learning.rl_agent import RLAgent
from src.reinforcement_learning.chess_rl_wrapper import ChessRLWrapper
from src.reinforcement_learning.checkers_rl_wrapper import CheckersRLWrapper

def test_rl_agent_creation():
    chess_env = ChessRLWrapper()
    chess_agent = RLAgent(state_size=3**64, action_size=len(chess_env.action_space))
    assert chess_agent is not None

    checkers_env = CheckersRLWrapper()
    checkers_agent = RLAgent(state_size=3**32, action_size=len(checkers_env.action_space))
    assert checkers_agent is not None

def test_rl_agent_action():
    chess_env = ChessRLWrapper()
    chess_agent = RLAgent(state_size=3**64, action_size=len(chess_env.action_space))
    state = chess_env.reset()
    action = chess_agent.get_action(state)
    assert 0 <= action < len(chess_env.action_space)

    checkers_env = CheckersRLWrapper()
    checkers_agent = RLAgent(state_size=3**32, action_size=len(checkers_env.action_space))
    state = checkers_env.reset()
    action = checkers_agent.get_action(state)
    assert 0 <= action < len(checkers_env.action_space)

def test_rl_agent_update():
    chess_env = ChessRLWrapper()
    chess_agent = RLAgent(state_size=3**64, action_size=len(chess_env.action_space))
    state = chess_env.reset()
    action = chess_agent.get_action(state)
    next_state, reward, done, _ = chess_env.step(action)
    chess_agent.update(state, action, reward, next_state, done)

    checkers_env = CheckersRLWrapper()
    checkers_agent = RLAgent(state_size=3**32, action_size=len(checkers_env.action_space))
    state = checkers_env.reset()
    action = checkers_agent.get_action(state)
    next_state, reward, done, _ = checkers_env.step(action)
    checkers_agent.update(state, action, reward, next_state, done)

if __name__ == "__main__":
    test_rl_agent_creation()
    test_rl_agent_action()
    test_rl_agent_update()
    print("All tests passed!")
