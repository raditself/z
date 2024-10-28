
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any
import numpy as np
from src.alphazero.mcts import MCTS
from src.alphazero.game import Game

class SearchTreeNode:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children: List[SearchTreeNode] = []
        self.visits = 0
        self.value = 0.0

    def add_child(self, child_state, child_action):
        child = SearchTreeNode(child_state, child_action, self)
        self.children.append(child)
        return child

class GameTrajectory:
    def __init__(self):
        self.states: List[Any] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = []

    def add_step(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

class AnalysisToolkit:
    def __init__(self, game: Game, mcts: MCTS):
        self.game = game
        self.mcts = mcts

    def extract_search_tree(self, root_state) -> SearchTreeNode:
        root = SearchTreeNode(root_state)
        self._build_tree(root, self.mcts.root)
        return root

    def _build_tree(self, analysis_node: SearchTreeNode, mcts_node):
        for action, child in mcts_node.children.items():
            child_state = self.game.get_next_state(analysis_node.state, action)
            child_node = analysis_node.add_child(child_state, action)
            child_node.visits = child.visit_count
            child_node.value = child.value()
            self._build_tree(child_node, child)

    def analyze_search_tree(self, root: SearchTreeNode) -> Dict[str, Any]:
        max_depth = self._get_max_depth(root)
        total_nodes = self._count_nodes(root)
        avg_branching_factor = self._get_avg_branching_factor(root)

        return {
            "max_depth": max_depth,
            "total_nodes": total_nodes,
            "avg_branching_factor": avg_branching_factor
        }

    def _get_max_depth(self, node: SearchTreeNode) -> int:
        if not node.children:
            return 0
        return 1 + max(self._get_max_depth(child) for child in node.children)

    def _count_nodes(self, node: SearchTreeNode) -> int:
        return 1 + sum(self._count_nodes(child) for child in node.children)

    def _get_avg_branching_factor(self, node: SearchTreeNode) -> float:
        if not node.children:
            return 0
        return np.mean([len(child.children) for child in node.children])

    def analyze_game_trajectory(self, trajectory: GameTrajectory) -> Dict[str, Any]:
        game_length = len(trajectory.states)
        total_reward = sum(trajectory.rewards)
        avg_reward = total_reward / game_length if game_length > 0 else 0

        return {
            "game_length": game_length,
            "total_reward": total_reward,
            "avg_reward": avg_reward
        }

    def visualize_search_tree(self, root: SearchTreeNode, max_depth: int = 3):
        G = nx.Graph()
        self._build_graph(G, root, 0, max_depth)
        
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(G, 'action')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title("Search Tree Visualization")
        plt.axis('off')

    def get_piece_values(self, nnet):
        # This is a placeholder implementation. You'll need to adapt this to your specific neural network architecture.
        piece_values = nnet.get_piece_values()
        return {
            'pawn': piece_values[0],
            'knight': piece_values[1],
            'bishop': piece_values[2],
            'rook': piece_values[3],
            'queen': piece_values[4]
        }

    def get_concept_importances(self, nnet):
        # This is a placeholder implementation. You'll need to adapt this to your specific neural network architecture.
        concept_importances = nnet.get_concept_importances()
        return {
            'material': concept_importances[0],
            'position': concept_importances[1],
            'king_safety': concept_importances[2],
            'pawn_structure': concept_importances[3]
        }

    def get_opening_preferences(self, nnet):
        # This is a placeholder implementation. You'll need to adapt this to your specific game representation.
        opening_prefs = nnet.get_opening_preferences()
        return {
            'e4': opening_prefs[0],
            'd4': opening_prefs[1],
            'c4': opening_prefs[2],
            'Nf3': opening_prefs[3]
        }

    def compare_iterations(self, iteration_data, current_iteration, writer):
        # Extract data for comparison
        iterations = [data['iteration'] for data in iteration_data]
        win_rates = [data['win_rate'] for data in iteration_data]
        losses = [data['loss'] for data in iteration_data]
        avg_game_lengths = [data['avg_game_length'] for data in iteration_data]
        
        # Plot win rates
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, win_rates, marker='o')
        plt.title("Win Rate Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Win Rate")
        plt.grid(True)
        writer.add_figure('Comparison/WinRate', plt.gcf(), current_iteration)
        plt.close()

        # Plot losses
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, losses, marker='o')
        plt.title("Loss Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        writer.add_figure('Comparison/Loss', plt.gcf(), current_iteration)
        plt.close()

        # Plot average game lengths
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, avg_game_lengths, marker='o')
        plt.title("Average Game Length Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Average Game Length")
        plt.grid(True)
        writer.add_figure('Comparison/AvgGameLength', plt.gcf(), current_iteration)
        plt.close()

        # Compare piece values
        latest_piece_values = iteration_data[-1]['piece_values']
        for piece, value in latest_piece_values.items():
            writer.add_scalar(f'PieceValues/{piece}', value, current_iteration)

        # Compare concept importances
        latest_concept_importances = iteration_data[-1]['concept_importances']
        for concept, importance in latest_concept_importances.items():
            writer.add_scalar(f'ConceptImportances/{concept}', importance, current_iteration)

        # Compare opening preferences
        latest_opening_prefs = iteration_data[-1]['opening_preferences']
        for opening, preference in latest_opening_prefs.items():
            writer.add_scalar(f'OpeningPreferences/{opening}', preference, current_iteration)

    def compare_training_iterations(self, iteration_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        comparison_results = {
            "win_rates": [],
            "avg_game_lengths": [],
            "avg_rewards": [],
            "max_depths": [],
            "total_nodes": [],
            "avg_branching_factors": []
        }

        for iteration in iteration_data:
            comparison_results["win_rates"].append(iteration.get("win_rate", 0))
            comparison_results["avg_game_lengths"].append(iteration.get("avg_game_length", 0))
            comparison_results["avg_rewards"].append(iteration.get("avg_reward", 0))
            comparison_results["max_depths"].append(iteration.get("max_depth", 0))
            comparison_results["total_nodes"].append(iteration.get("total_nodes", 0))
            comparison_results["avg_branching_factors"].append(iteration.get("avg_branching_factor", 0))

        return comparison_results

    def visualize_training_comparison(self, comparison_results: Dict[str, List[float]]):
        metrics = list(comparison_results.keys())
        num_metrics = len(metrics)
        num_iterations = len(comparison_results[metrics[0]])

        fig, axs = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))
        fig.suptitle("Training Iteration Comparison")

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, num_iterations + 1), comparison_results[metric])
            axs[i].set_xlabel("Iteration")
            axs[i].set_ylabel(metric.replace("_", " ").title())
            axs[i].grid(True)

        plt.tight_layout()
        plt.show()
        plt.tight_layout()
        plt.savefig('search_tree_visualization.png')
        plt.close()
        print("Search tree visualization saved as 'search_tree_visualization.png'")

    def _build_graph(self, G: nx.Graph, node: SearchTreeNode, depth: int, max_depth: int):
        if depth > max_depth:
            return
        
        G.add_node(id(node), label=f"V:{node.visits}\nQ:{node.value:.2f}")
        
        for child in node.children:
            G.add_node(id(child), label=f"V:{child.visits}\nQ:{child.value:.2f}")
            G.add_edge(id(node), id(child), action=str(child.action))
            self._build_graph(G, child, depth + 1, max_depth)

    def visualize_game_trajectory(self, trajectory: GameTrajectory):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(trajectory.rewards, marker='o')
        plt.title("Rewards over Game Trajectory")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        
        plt.subplot(2, 1, 2)
        cumulative_rewards = np.cumsum(trajectory.rewards)
        plt.plot(cumulative_rewards, marker='o')
        plt.title("Cumulative Rewards over Game Trajectory")
        plt.xlabel("Step")
        plt.ylabel("Cumulative Reward")
        
        plt.tight_layout()
        plt.savefig('game_trajectory_visualization.png')
        plt.close()
        print("Game trajectory visualization saved as 'game_trajectory_visualization.png'")

def create_sample_trajectory():
    trajectory = GameTrajectory()
    for i in range(10):
        state = f"State_{i}"
        action = f"Action_{i}"
        reward = np.random.uniform(-1, 1)
        trajectory.add_step(state, action, reward)
    return trajectory

def create_sample_search_tree():
    root = SearchTreeNode("Root")
    for i in range(3):
        child = root.add_child(f"Child_{i}", f"Action_{i}")
        child.visits = np.random.randint(1, 100)
        child.value = np.random.uniform(0, 1)
        for j in range(2):
            grandchild = child.add_child(f"GrandChild_{i}_{j}", f"Action_{i}_{j}")
            grandchild.visits = np.random.randint(1, 50)
            grandchild.value = np.random.uniform(0, 1)
    return root

if __name__ == "__main__":
    # Test the visualization functions
    class DummyGame(Game):
        def get_next_state(self, state, action):
            return f"{state}_{action}"

    class DummyMCTS(MCTS):
        pass

    game = DummyGame()
    mcts = DummyMCTS(game)
    
    toolkit = AnalysisToolkit(game, mcts)
    
    # Test search tree visualization
    sample_tree = create_sample_search_tree()
    toolkit.visualize_search_tree(sample_tree)
    
    # Test game trajectory visualization
    sample_trajectory = create_sample_trajectory()
    toolkit.visualize_game_trajectory(sample_trajectory)
