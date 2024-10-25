import torch
import torch.nn as nn
from neural_architecture_search import create_model

class ChessModel(nn.Module):
    def __init__(self, model_path=None):
        super(ChessModel, self).__init__()
        if model_path:
            self.load_best_model(model_path)
        else:
            # Default architecture if no best model is available
            self.model = create_model(num_layers=5, neurons_per_layer=256, activation_func=nn.ReLU, dropout_rate=0.3)

    def forward(self, x):
        return self.model(x)

    def load_best_model(self, model_path):
        state_dict = torch.load(model_path)
        self.model = create_model(
            num_layers=len([name for name in state_dict.keys() if 'weight' in name]) - 1,
            neurons_per_layer=state_dict['0.weight'].size(0),
            activation_func=nn.ReLU,  # Assuming ReLU for simplicity
            dropout_rate=0.3  # Assuming a default dropout rate
        )
        self.model.load_state_dict(state_dict)

def get_model(model_path=None):
    return ChessModel(model_path)

