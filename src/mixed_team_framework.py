
import numpy as np
from src.mcts import MCTS
from src.neural_network import NeuralNetwork
from src.uncertainty_estimator import UncertaintyEstimator

class MixedTeamFramework:
    def __init__(self, mcts: MCTS, neural_network: NeuralNetwork, uncertainty_estimator: UncertaintyEstimator):
        self.mcts = mcts
        self.neural_network = neural_network
        self.uncertainty_estimator = uncertainty_estimator

    def get_ai_move(self, state, temperature=1.0):
        # Use MCTS to get the best move
        mcts_policy = self.mcts.get_action_prob(state, temperature)
        return np.argmax(mcts_policy)

    def get_human_move(self, state, valid_moves):
        # This method should be implemented by the user interface
        # Here, we'll just return a random valid move as a placeholder
        return np.random.choice(valid_moves)

    def evaluate_human_move(self, state, move):
        policy, value = self.neural_network.predict(state)
        move_strength = policy[move]
        uncertainty = self.uncertainty_estimator.estimate_uncertainty(state)

        evaluation = {
            'move_strength': move_strength,
            'value': value,
            'uncertainty': uncertainty
        }

        return evaluation

    def suggest_move(self, state, valid_moves, num_suggestions=3):
        mcts_policy = self.mcts.get_action_prob(state, temperature=1.0)
        sorted_moves = np.argsort(mcts_policy)[::-1]
        
        suggestions = []
        for move in sorted_moves:
            if move in valid_moves:
                evaluation = self.evaluate_human_move(state, move)
                suggestions.append((move, evaluation))
                if len(suggestions) == num_suggestions:
                    break

        return suggestions

    def play_mixed_team_game(self, game, human_team, ai_team, max_moves=1000):
        state = game.get_initial_state()
        move_history = []

        for _ in range(max_moves):
            if game.is_game_over(state):
                break

            current_team = game.get_current_team(state)
            valid_moves = game.get_valid_moves(state)

            if current_team in human_team:
                # Human turn
                suggestions = self.suggest_move(state, valid_moves)
                move = self.get_human_move(state, valid_moves)
                evaluation = self.evaluate_human_move(state, move)
            else:
                # AI turn
                move = self.get_ai_move(state)
                evaluation = self.evaluate_human_move(state, move)

            state = game.get_next_state(state, move)
            move_history.append((move, evaluation))

        winner = game.get_winner(state)
        return state, move_history, winner

    def analyze_mixed_team_game(self, game, move_history, winner):
        analysis = []
        for i, (move, evaluation) in enumerate(move_history):
            analysis.append(f"Move {i+1}:")
            analysis.append(f"  Strength: {evaluation['move_strength']:.2f}")
            analysis.append(f"  Value: {evaluation['value']:.2f}")
            analysis.append(f"  Uncertainty: {evaluation['uncertainty']['entropy']:.2f}")

        if winner == 1:
            analysis.append("The mixed human-AI team won the game.")
        elif winner == -1:
            analysis.append("The mixed human-AI team lost the game.")
        else:
            analysis.append("The game ended in a draw.")

        return "\n".join(analysis)
