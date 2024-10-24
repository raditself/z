
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size, action_size, num_channels=256):
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size
        self.action_size = action_size
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(6, num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels)

        self.pi_conv = nn.Conv2d(num_channels, 2, 1)
        self.pi_fc = nn.Linear(2 * board_size * board_size, action_size)

        self.v_conv = nn.Conv2d(num_channels, 1, 1)
        self.v_fc1 = nn.Linear(board_size * board_size, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, s):
        s = s.view(-1, 6, self.board_size, self.board_size)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))

        pi = F.relu(self.pi_conv(s))
        pi = pi.view(-1, 2 * self.board_size * self.board_size)
        pi = self.pi_fc(pi)
        pi = F.log_softmax(pi, dim=1)

        v = F.relu(self.v_conv(s))
        v = v.view(-1, self.board_size * self.board_size)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))

        return pi, v

def create_model(board_size=8, action_size=4096):
    return AlphaZeroNet(board_size, action_size)
