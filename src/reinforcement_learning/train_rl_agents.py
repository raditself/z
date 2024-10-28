
from rl_agent import RLAgent, train_agent
from chess_rl_wrapper import ChessRLWrapper
from checkers_rl_wrapper import CheckersRLWrapper

def train_chess_agent(num_episodes=10000):
    print("Training Chess Agent...")
    chess_env = ChessRLWrapper()
    chess_agent = RLAgent(state_size=3**64, action_size=len(chess_env.action_space))
    train_agent(chess_agent, chess_env, num_episodes)
    return chess_agent

def train_checkers_agent(num_episodes=10000):
    print("Training Checkers Agent...")
    checkers_env = CheckersRLWrapper()
    checkers_agent = RLAgent(state_size=3**32, action_size=len(checkers_env.action_space))
    train_agent(checkers_agent, checkers_env, num_episodes)
    return checkers_agent

def test_agent(agent, env, num_games=100):
    wins = 0
    for _ in range(num_games):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
        if reward > 0:
            wins += 1
    return wins / num_games

if __name__ == "__main__":
    # Train and test Chess agent
    chess_agent = train_chess_agent()
    chess_env = ChessRLWrapper()
    chess_win_rate = test_agent(chess_agent, chess_env)
    print(f"Chess Agent Win Rate: {chess_win_rate:.2f}")

    # Train and test Checkers agent
    checkers_agent = train_checkers_agent()
    checkers_env = CheckersRLWrapper()
    checkers_win_rate = test_agent(checkers_agent, checkers_env)
    print(f"Checkers Agent Win Rate: {checkers_win_rate:.2f}")
