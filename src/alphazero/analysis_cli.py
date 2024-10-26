
import argparse
import numpy as np
from src.alphazero.analysis_toolkit import AnalysisToolkit, create_sample_search_tree, create_sample_trajectory
from src.alphazero.game import Game
from src.alphazero.mcts import MCTS

class DummyGame(Game):
    def __init__(self):
        self.board_size = (3, 3)
        self.action_size = 9

    def get_initial_state(self):
        return [0] * 9

    def get_next_state(self, state, action):
        new_state = state.copy()
        new_state[action] = 1
        return new_state

    def get_valid_moves(self, state):
        return [i for i, v in enumerate(state) if v == 0]

    def get_value_and_terminated(self, state, action):
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_action_size(self):
        return self.action_size

    def get_board_size(self):
        return self.board_size

    def get_canonical_form(self, state, player):
        return state

class DummyModel:
    def __init__(self, game):
        self.game = game

    def predict(self, state):
        policy = np.ones(self.game.get_action_size()) / self.game.get_action_size()
        value = 0
        return policy, value

class DummyMCTS(MCTS):
    def __init__(self, game):
        model = DummyModel(game)
        super().__init__(game, model)
        self.root = None

def main():
    parser = argparse.ArgumentParser(description="AlphaZero Analysis Toolkit CLI")
    parser.add_argument("--mode", choices=["tree", "trajectory"], required=True, help="Analysis mode: tree or trajectory")
    args = parser.parse_args()

    game = DummyGame()
    mcts = DummyMCTS(game)
    toolkit = AnalysisToolkit(game, mcts)

    if args.mode == "tree":
        analyze_search_tree(toolkit)
    elif args.mode == "trajectory":
        analyze_game_trajectory(toolkit)

def analyze_search_tree(toolkit):
    print("Analyzing search tree...")
    sample_tree = create_sample_search_tree()
    
    # Perform analysis
    analysis_results = toolkit.analyze_search_tree(sample_tree)
    print("Analysis results:")
    for key, value in analysis_results.items():
        print(f"{key}: {value}")
    
    # Visualize search tree
    print("\nVisualizing search tree...")
    toolkit.visualize_search_tree(sample_tree)

def analyze_game_trajectory(toolkit):
    print("Analyzing game trajectory...")
    sample_trajectory = create_sample_trajectory()
    
    # Perform analysis
    analysis_results = toolkit.analyze_game_trajectory(sample_trajectory)
    print("Analysis results:")
    for key, value in analysis_results.items():
        print(f"{key}: {value}")
    
    # Visualize game trajectory
    print("\nVisualizing game trajectory...")
    toolkit.visualize_game_trajectory(sample_trajectory)

if __name__ == "__main__":
    main()
