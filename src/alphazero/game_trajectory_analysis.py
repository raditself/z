
import matplotlib.pyplot as plt
from src.alphazero.game_analysis import GameAnalysis
import chess.pgn
import io

class GameTrajectoryAnalysis(GameAnalysis):
    def __init__(self, model_path):
        super().__init__(model_path)

    def visualize_evaluation_trajectory(self, analysis):
        move_numbers = list(range(1, len(analysis) + 1))
        evaluations = [move['evaluation'] for move in analysis]

        plt.figure(figsize=(12, 6))
        plt.plot(move_numbers, evaluations, marker='o')
        plt.title('Game Evaluation Trajectory')
        plt.xlabel('Move Number')
        plt.ylabel('Evaluation')
        plt.grid(True)
        plt.savefig('evaluation_trajectory.png')
        plt.close()

    def visualize_blunders(self, blunders, analysis):
        move_numbers = list(range(1, len(analysis) + 1))
        evaluations = [move['evaluation'] for move in analysis]

        plt.figure(figsize=(12, 6))
        plt.plot(move_numbers, evaluations, marker='o')
        
        for blunder in blunders:
            plt.plot(blunder['move_number'], blunder['curr_eval'], 'ro', markersize=10)
        
        plt.title('Game Evaluation Trajectory with Blunders')
        plt.xlabel('Move Number')
        plt.ylabel('Evaluation')
        plt.grid(True)
        plt.savefig('blunders_visualization.png')
        plt.close()

    def analyze_and_visualize_game(self, pgn_string, depth=20):
        game = chess.pgn.read_game(io.StringIO(pgn_string))
        analysis = self.analyze_game(game, depth)
        self.visualize_evaluation_trajectory(analysis)
        
        blunders = self.find_blunders(analysis)
        self.visualize_blunders(blunders, analysis)
        
        return analysis, blunders

# Usage example:
# analyzer = GameTrajectoryAnalysis('path_to_chess_model.h5')
# pgn_string = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O"
# analysis, blunders = analyzer.analyze_and_visualize_game(pgn_string)
# print("Analysis:", analysis)
# print("Blunders:", blunders)
