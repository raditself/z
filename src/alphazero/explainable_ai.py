
import torch
import numpy as np
import shap

class ExplainableAI:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.DeepExplainer(self.model, torch.zeros((1, 3, 8, 8)))

    def explain_decision(self, state, action):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        shap_values = self.explainer.shap_values(state_tensor)
        
        # Get the SHAP values for the chosen action
        action_shap = shap_values[action]

        # Compute the importance of each square for the decision
        importance_map = np.sum(np.abs(action_shap), axis=1)

        # Find the top 3 most important squares
        top_squares = np.argsort(importance_map.flatten())[-3:]
        
        explanation = "The model's decision was mainly influenced by:
"
        for square in top_squares:
            row, col = square // 8, square % 8
            explanation += f"- Square ({row}, {col}) with importance {importance_map[row, col]:.2f}
"

        return explanation

    def visualize_explanation(self, state, action):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        shap_values = self.explainer.shap_values(state_tensor)
        
        shap.image_plot(shap_values, -state_tensor.numpy())

if __name__ == "__main__":
    from src.alphazero.neural_network import DynamicNeuralNetwork
    from src.games.chess import Chess

    game = Chess()
    model = DynamicNeuralNetwork(game)
    explainer = ExplainableAI(model)

    state = game.get_initial_state()
    action = 0  # Example action

    explanation = explainer.explain_decision(state, action)
    print(explanation)

    explainer.visualize_explanation(state, action)
