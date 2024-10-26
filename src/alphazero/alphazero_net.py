
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, num_hidden):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.relu = nn.LeakyReLU(inplace=True)
        self.se = SEBlock(num_hidden)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class AlphaZeroNet(nn.Module):
    def __init__(self, game_size, action_size, num_resblocks=19, num_hidden=256):
        super(AlphaZeroNet, self).__init__()
        self.game_size = game_size
        self.action_size = action_size
        self.num_hidden = num_hidden

        self.conv1 = nn.Conv2d(3, num_hidden, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.relu = nn.LeakyReLU(inplace=True)

        self.resblocks = nn.Sequential(*[ResidualBlock(num_hidden) for _ in range(num_resblocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * game_size * game_size, action_size)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, 1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.resblocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
