
import numpy as np
from src.mcts import MCTS
from src.neural_network import NeuralNetwork
from src.uncertainty_estimator import UncertaintyEstimator

class CommentaryGenerator:
    def __init__(self, mcts: MCTS, neural_network: NeuralNetwork, uncertainty_estimator: UncertaintyEstimator):
        self.mcts = mcts
        self.neural_network = neural_network
        self.uncertainty_estimator = uncertainty_estimator

    def generate_move_commentary(self, state, move, next_state):
        policy, value = self.neural_network.predict(state)
        next_policy, next_value = self.neural_network.predict(next_state)
        uncertainty = self.uncertainty_estimator.estimate_uncertainty(state)
        
        move_strength = policy[move]
        value_change = next_value - value
        
        commentary = []
        
        # Comment on move strength
        if move_strength > 0.8:
            commentary.append("This appears to be a very strong move.")
        elif move_strength > 0.6:
            commentary.append("This seems to be a good move.")
        elif move_strength < 0.2:
            commentary.append("This move is unexpected and might be suboptimal.")
        
        # Comment on value change
        if value_change > 0.2:
            commentary.append("This move significantly improves the position.")
        elif value_change < -0.2:
            commentary.append("This move appears to weaken the position.")
        
        # Comment on uncertainty
        if uncertainty['entropy'] > 0.5:
            commentary.append("There's a high degree of uncertainty in this position.")
        elif uncertainty['entropy'] < 0.1:
            commentary.append("The evaluation of this position is quite certain.")
        
        return " ".join(commentary)

    def generate_game_commentary(self, game_states, moves):
        full_commentary = []
        for i, (state, move) in enumerate(zip(game_states[:-1], moves)):
            next_state = game_states[i+1]
            move_commentary = self.generate_move_commentary(state, move, next_state)
            full_commentary.append(f"Move {i+1}: {move_commentary}")
        
        return "\n".join(full_commentary)

    def analyze_game_outcome(self, final_state, winner):
        _, final_value = self.neural_network.predict(final_state)
        
        if winner == 1:
            return f"AlphaZero won the game. Final position evaluation: {final_value:.2f}"
        elif winner == -1:
            return f"AlphaZero lost the game. Final position evaluation: {final_value:.2f}"
        else:
            return f"The game ended in a draw. Final position evaluation: {final_value:.2f}"
