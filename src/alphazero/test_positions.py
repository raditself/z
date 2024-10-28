
import chess
from typing import List, Tuple

def create_test_positions() -> List[Tuple[chess.Board, str, float]]:
    """
    Create a list of test positions with their descriptions and expected evaluation scores.
    
    Returns:
        List of tuples containing (chess.Board, description, expected_score)
    """
    positions = []

    # Opening theory position
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    positions.append((board, "King's Pawn Opening", 0.3))

    # Middlegame tactical position
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5")
    positions.append((board, "Italian Game - Giuoco Piano", 0.1))

    # Endgame position
    board = chess.Board("4k3/8/4P3/3K4/8/8/8/8 w - - 0 1")
    positions.append((board, "King and Pawn vs King Endgame", 0.9))

    # Complex middlegame position
    board = chess.Board("r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1B3/PPPQ1PPP/2KR1B1R w - - 0 10")
    positions.append((board, "Complex Middlegame", 0.0))

    # Attacking position
    board = chess.Board("r1bqk2r/ppp2ppp/2n1pn2/3p4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 0 7")
    positions.append((board, "Attacking Position", 0.4))

    # Defensive position
    board = chess.Board("rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6")
    positions.append((board, "Defensive Position", -0.2))

    # Pawn structure test
    board = chess.Board("r1bqkbnr/pp2pppp/2n5/2pp4/3P4/2N2N2/PPP1PPPP/R1BQKB1R w KQkq - 0 5")
    positions.append((board, "Isolated Queen's Pawn", 0.1))

    # Knight outpost
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/3P4/2N2N2/PPP1PPPP/R1BQKB1R w KQkq - 0 5")
    positions.append((board, "Knight Outpost Opportunity", 0.2))

    # Opposite-colored bishops endgame
    board = chess.Board("2k5/8/8/2b5/8/5B2/8/2K5 w - - 0 1")
    positions.append((board, "Opposite-Colored Bishops Endgame", 0.0))

    # Rook endgame
    board = chess.Board("8/8/8/8/3k4/8/4K3/4R3 w - - 0 1")
    positions.append((board, "Rook vs King Endgame", 0.7))

    return positions

def evaluate_alphazero_on_positions(alphazero_model, positions: List[Tuple[chess.Board, str, float]]) -> List[Tuple[str, float, float]]:
    """
    Evaluate AlphaZero's performance on the given test positions.
    
    Args:
        alphazero_model: The AlphaZero model to evaluate
        positions: List of tuples containing (chess.Board, description, expected_score)
    
    Returns:
        List of tuples containing (description, expected_score, alphazero_score)
    """
    results = []
    for board, description, expected_score in positions:
        alphazero_score = alphazero_model.evaluate_position(board)
        results.append((description, expected_score, alphazero_score))
    return results

def print_evaluation_results(results: List[Tuple[str, float, float]]):
    """
    Print the evaluation results in a formatted manner.
    
    Args:
        results: List of tuples containing (description, expected_score, alphazero_score)
    """
    print("AlphaZero Test Position Evaluation Results")
    print("==========================================")
    print(f"{'Position':<30} {'Expected':<10} {'AlphaZero':<10} {'Difference':<10}")
    print("-" * 60)
    
    for description, expected_score, alphazero_score in results:
        difference = alphazero_score - expected_score
        print(f"{description:<30} {expected_score:<10.2f} {alphazero_score:<10.2f} {difference:<10.2f}")

if __name__ == "__main__":
    # This is a placeholder for when we have the actual AlphaZero model
    class DummyAlphaZero:
        def evaluate_position(self, board):
            return 0.5  # Dummy evaluation
    
    alphazero_model = DummyAlphaZero()
    test_positions = create_test_positions()
    results = evaluate_alphazero_on_positions(alphazero_model, test_positions)
    print_evaluation_results(results)
