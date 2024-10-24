
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
