
import numpy as np
from src.mcts import MCTS
from src.neural_network import NeuralNetwork

class UncertaintyEstimator:
    def __init__(self, mcts: MCTS, neural_network: NeuralNetwork):
        self.mcts = mcts
        self.neural_network = neural_network

    def estimate_uncertainty(self, state):
        # Get the MCTS visit counts and value estimates
        visit_counts = self.mcts.get_visit_counts(state)
        value_estimates = self.mcts.get_value_estimates(state)

        # Calculate the entropy of the visit counts
        total_visits = np.sum(visit_counts)
        probabilities = visit_counts / total_visits
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))

        # Get the neural network's policy and value predictions
        policy, value = self.neural_network.predict(state)

        # Calculate the KL divergence between MCTS policy and neural network policy
        kl_divergence = np.sum(probabilities * np.log((probabilities + 1e-8) / (policy + 1e-8)))

        # Calculate the variance of value estimates
        value_variance = np.var(value_estimates)

        # Combine the uncertainty measures
        uncertainty = {
            'entropy': entropy,
            'kl_divergence': kl_divergence,
            'value_variance': value_variance
        }

        return uncertainty

    def get_uncertainty_weighted_policy(self, state, temperature=1.0):
        uncertainty = self.estimate_uncertainty(state)
        visit_counts = self.mcts.get_visit_counts(state)
        
        # Apply temperature and uncertainty weighting
        adjusted_counts = visit_counts ** (1 / temperature)
        uncertainty_weight = 1 / (1 + uncertainty['entropy'])
        weighted_counts = adjusted_counts * uncertainty_weight
        
        policy = weighted_counts / np.sum(weighted_counts)
        return policy
