
import torch
import os
import json


import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessModel(nn.Module):
    def __init__(self, board_size=8, action_size=4672):  # 4672 is an estimate of possible chess moves
        super(ChessModel, self).__init__()
        self.board_size = board_size
        self.action_size = action_size
        
        # Common layers
        self.conv1 = nn.Conv2d(6, 256, 3, padding=1)  # 6 input channels for each piece type
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, 1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(256, 1, 1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Common layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

    def predict(self, state):
        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get policy and value predictions
        policy, value = self(state_tensor)
        
        # Convert to numpy arrays and remove batch dimension
        policy = policy.detach().numpy()[0]
        value = value.detach().numpy()[0][0]
        
        return policy, value


def export_model(model, file_path):
    """
    Export the model to a file.
    """
    model_state = model.state_dict()
    torch.save(model_state, file_path)
    print("Model exported to " + file_path)

def import_model(model, file_path):
    """
    Import the model from a file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Model file not found: " + file_path)
    
    model_state = torch.load(file_path)
    model.load_state_dict(model_state)
    print("Model imported from " + file_path)

def export_model_metadata(model, file_path):
    """
    Export model metadata (architecture, hyperparameters) to a JSON file.
    """
    metadata = {
        "architecture": str(model),
        "num_layers": model.num_layers,
        "num_channels": model.num_channels,
        # Add any other relevant metadata
    }
    
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Model metadata exported to " + file_path)

def import_model_metadata(file_path):
    """
    Import model metadata from a JSON file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Metadata file not found: " + file_path)
    
    with open(file_path, 'r') as f:
        metadata = json.load(f)
    return metadata
