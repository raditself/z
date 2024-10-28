
import chess
from src.alphazero.chess_ai import ChessAI
from src.alphazero.mcts import MCTS, get_action_probs

class GameAnalysis:
    def __init__(self, model_path):
        self.chess_ai = ChessAI(model_path)
        self.mcts = MCTS(self.chess_ai.model)

    def analyze_position(self, board, num_moves=5, num_simulations=800):
        root = self.mcts.search(board, num_simulations)
        actions, probs = get_action_probs(root, temperature=0)
        
        top_moves = sorted(zip(actions, probs), key=lambda x: x[1], reverse=True)[:num_moves]
        
        analysis = []
        for move, prob in top_moves:
            next_board = board.copy()
            next_board.push(move)
            _, value = self.chess_ai.get_move(next_board.fen(), num_simulations=100)
            
            analysis.append({
                'move': move,
                'probability': prob,
                'evaluation': value
            })
        
        return analysis

    def analyze_game(self, game, num_moves=5, num_simulations=800):
        board = game.board()
        analysis = []
        
        for move in game.mainline_moves():
            position_analysis = self.analyze_position(board, num_moves, num_simulations)
            analysis.append({
                'fen': board.fen(),
                'move_played': move,
                'top_moves': position_analysis
            })
            board.push(move)
        
        return analysis

# Example usage:
# import chess.pgn
# game_analyzer = GameAnalysis('path_to_model_weights.h5')
# pgn = open('game.pgn')
# game = chess.pgn.read_game(pgn)
# analysis = game_analyzer.analyze_game(game)
# for position in analysis:
#     print(f"Position: {position['fen']}")
#     print(f"Move played: {position['move_played']}")
#     print("Top alternative moves:")
#     for move in position['top_moves']:
#         print(f"  {move['move']}: prob={move['probability']:.3f}, eval={move['evaluation']:.3f}")
#     print()
