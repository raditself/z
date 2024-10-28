
import chess
import time
from src.alphazero.mcts import HierarchicalMCTS

def test_dynamic_branching():
    board = chess.Board()
    mcts = HierarchicalMCTS(max_branching_factor=30, min_branching_factor=5, num_simulations=1000)

    def run_test(board, time_limit, description):
        print(f"Testing with {description}")
        print(f"Time limit: {time_limit} seconds")
        print(f"FEN: {board.fen()}")
        
        start_time = time.time()
        best_move = mcts.search(board, time_limit=time_limit)
        end_time = time.time()
        
        root_node = mcts.root
        print(f"Number of children explored: {len(root_node.children)}")
        print(f"Total visit count: {root_node.visit_count}")
        print(f"Dynamic branching factor: {mcts.calculate_dynamic_branching_factor(board)}")
        print(f"Position complexity: {mcts.calculate_position_complexity(board):.2f}")
        print(f"Best move: {best_move}")
        print(f"Actual time taken: {end_time - start_time:.2f} seconds")
        print(f"Top 5 moves:")
        sorted_moves = sorted(root_node.children.items(), key=lambda x: x[1].visit_count, reverse=True)[:5]
        for move, node in sorted_moves:
            print(f"  {move}: visits={node.visit_count}, value={node.value():.3f}")
        print()

    # Test 1: Initial position
    run_test(board, 1, "initial position")

    # Test 2: Complex middlegame position
    complex_fen = "r1bq1rk1/pp2ppbp/2np1np1/2p5/2B1P3/2NP1N2/PPP1QPPP/R1B2RK1 w - - 0 9"
    board.set_fen(complex_fen)
    run_test(board, 3, "complex middlegame position")

    # Test 3: Endgame position
    endgame_fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    board.set_fen(endgame_fen)
    run_test(board, 2, "simple endgame position")

    # Test 4: Tactical position
    tactical_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    board.set_fen(tactical_fen)
    run_test(board, 5, "tactical position")

if __name__ == "__main__":
    test_dynamic_branching()
