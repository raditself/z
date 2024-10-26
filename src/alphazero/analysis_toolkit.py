
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
