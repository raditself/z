import time
import chess
from src.alphazero.mcts import MCTS
from src.alphazero.model import AlphaZeroModel

def run_benchmark(mcts, num_simulations, num_games):
    total_time = 0
    total_nodes = 0

    for game in range(num_games):
        print(f"Running game {game + 1}/{num_games}")
        board = chess.Board()
        start_time = time.time()
        try:
            root = mcts.search(board, num_simulations)
            end_time = time.time()
            game_time = end_time - start_time
            nodes = count_nodes(root)
            total_time += game_time
            total_nodes += nodes
            print(f"  Time: {game_time:.2f} seconds")
            print(f"  Nodes: {nodes}")
        except Exception as e:
            print(f"  Error: {str(e)}")

    if num_games > 0:
        avg_time = total_time / num_games
        avg_nodes = total_nodes / num_games
    else:
        avg_time = 0
        avg_nodes = 0

    return avg_time, avg_nodes

def count_nodes(node):
    count = 1
    for child in node.children.values():
        count += count_nodes(child)
    return count

def main():
    model = AlphaZeroModel()  # You might need to adjust this based on your actual model implementation
    num_simulations = 100  # Reduced from 1000
    num_games = 3  # Reduced from 10

    print("Running benchmark for Original MCTS")
    original_mcts = MCTS(model, use_rave=False, use_progressive_widening=False)
    original_time, original_nodes = run_benchmark(original_mcts, num_simulations, num_games)

    print("\nRunning benchmark for Enhanced MCTS")
    enhanced_mcts = MCTS(model, use_rave=True, use_progressive_widening=True)
    enhanced_time, enhanced_nodes = run_benchmark(enhanced_mcts, num_simulations, num_games)

    print(f"\nResults:")
    print(f"Original MCTS:")
    print(f"  Average time per game: {original_time:.2f} seconds")
    print(f"  Average nodes explored: {original_nodes:.2f}")
    print(f"Enhanced MCTS:")
    print(f"  Average time per game: {enhanced_time:.2f} seconds")
    print(f"  Average nodes explored: {enhanced_nodes:.2f}")
    print(f"Improvement:")
    print(f"  Time: {(original_time - enhanced_time) / original_time * 100:.2f}%")
    print(f"  Nodes: {(original_nodes - enhanced_nodes) / original_nodes * 100:.2f}%")

if __name__ == "__main__":
    main()
