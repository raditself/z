import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessModel(nn.Module):
    def __init__(self, input_shape=(8, 8, 14), num_filters=256, num_res_blocks=19):
        super(ChessModel, self).__init__()
        self.input_shape = input_shape
        self.conv = nn.Conv2d(input_shape[2], num_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * input_shape[0] * input_shape[1], 4672)  # 4672 is the number of possible moves in chess
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_shape[0] * input_shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        for res_block in self.res_blocks:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, policy, value = self.data[idx]
        return torch.FloatTensor(board), torch.FloatTensor(policy), torch.FloatTensor([value])

def get_model(model_path=None):
    model = ChessModel()
    if model_path:
        model.load_state_dict(torch.load(model_path))
    return model

def get_data_loader(data, batch_size=32, shuffle=True):
    dataset = ChessDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

