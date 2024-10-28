
import chess
import chess.pgn
import io
from src.alphazero.chess_ai import ChessAI

class GameAnalysis:
    def __init__(self, model_path):
        self.ai = ChessAI(model_path)

    def load_pgn(self, pgn_file):
        with open(pgn_file) as f:
            return chess.pgn.read_game(f)

    def analyze_game(self, game, depth=20):
        board = game.board()
        analysis = []
        for move in game.mainline_moves():
            board.push(move)
            _, evaluation = self.ai.get_move(board.fen(), temperature=0)
            analysis.append({
                'move': move,
                'evaluation': float(evaluation)  # Convert to float
            })
        return analysis

    def find_blunders(self, analysis, threshold=0.5):
        blunders = []
        for i in range(1, len(analysis)):
            prev_eval = analysis[i-1]['evaluation']
            curr_eval = analysis[i]['evaluation']
            if abs(curr_eval - prev_eval) > threshold:
                blunders.append({
                    'move_number': i + 1,
                    'move': analysis[i]['move'],
                    'prev_eval': prev_eval,
                    'curr_eval': curr_eval
                })
        return blunders

    def print_analysis(self, analysis):
        for i, move_analysis in enumerate(analysis):
            print(f"Move {i+1}: {move_analysis['move']} (Evaluation: {move_analysis['evaluation']:.2f})")

    def print_blunders(self, blunders):
        for blunder in blunders:
            print(f"Blunder at move {blunder['move_number']}: {blunder['move']}")
            print(f"Evaluation change: {blunder['prev_eval']:.2f} -> {blunder['curr_eval']:.2f}")

# Example usage:
# analyzer = GameAnalysis('path_to_chess_model.h5')
# game = analyzer.load_pgn('example_game.pgn')
# analysis = analyzer.analyze_game(game)
# analyzer.print_analysis(analysis)
# blunders = analyzer.find_blunders(analysis)
# analyzer.print_blunders(blunders)
