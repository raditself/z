
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DynamicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, game_phase):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x) * (1 + 0.2 * game_phase)  # Increase activation based on game phase

class AlphaZeroNet(nn.Module):
    def __init__(self, game_size, action_size, num_channels=256):
        super(AlphaZeroNet, self).__init__()
        self.conv1 = DynamicConvBlock(3, num_channels)
        self.conv2 = DynamicConvBlock(num_channels, num_channels)
        self.conv3 = DynamicConvBlock(num_channels, num_channels)
        self.conv4 = DynamicConvBlock(num_channels, num_channels)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

        self.fc1 = nn.Linear(num_channels * game_size * game_size, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, action_size)
        
        self.fc4 = nn.Linear(512, 1)

        self.game_size = game_size

    def forward(self, s, game_phase):
        s = s.view(-1, 3, self.game_size, self.game_size)
        s = self.conv1(s, game_phase)
        s = self.conv2(s, game_phase)
        s = self.conv3(s, game_phase)
        s = self.conv4(s, game_phase)

        s = s.view(-1, self.game_size * self.game_size * 256)

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=0.3, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=0.3, training=self.training)

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def prepare_for_quantization(self):
        # Prepare the model for quantization
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self, inplace=True)
