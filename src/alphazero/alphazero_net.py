
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    def __init__(self, game_size, action_size, num_resblocks=19, num_hidden=256):
        super(AlphaZeroNet, self).__init__()
        self.game_size = game_size
        self.action_size = action_size
        self.num_hidden = num_hidden

        self.conv1 = nn.Conv2d(3, num_hidden, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)

        self.resblocks = nn.ModuleList([self._build_residual_block(num_hidden) for _ in range(num_resblocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game_size * game_size, action_size)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game_size * game_size, 1),
            nn.Tanh()
        )

    def _build_residual_block(self, num_hidden):
        return nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_hidden)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        for block in self.resblocks:
            residual = x
            x = F.relu(x + block(x))

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value
