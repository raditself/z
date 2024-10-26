
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

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


import torch.nn.utils.prune as prune

class AlphaZeroNet(nn.Module):
    # ... (existing methods remain unchanged) ...

    def prune_model(self, amount=0.2):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')

    def get_model_sparsity(self):
        total_params = 0
        nonzero_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            nonzero_params += torch.nonzero(param).size(0)
        return (1 - nonzero_params / total_params) * 100

def prune_and_retrain(model, examples, batch_size, epochs, prune_amount=0.2):
    print(f"Initial model sparsity: {model.get_model_sparsity():.2f}%")
    model.prune_model(amount=prune_amount)
    print(f"Model sparsity after pruning: {model.get_model_sparsity():.2f}%")
    model.train_network(examples, batch_size, epochs)
    print(f"Final model sparsity: {model.get_model_sparsity():.2f}%")

