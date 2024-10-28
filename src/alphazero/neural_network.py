
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class NeuralNetwork(nn.Module):
    def __init__(self, game, num_resblocks=19, num_hidden=256):
        super(NeuralNetwork, self).__init__()
        self.game = game
        self.num_actions = game.action_size
        self.conv = nn.Conv2d(3, num_hidden, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_hidden)
        self.resblocks = nn.ModuleList([ResidualBlock(num_hidden) for _ in range(num_resblocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.board_size * game.board_size, self.num_actions)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.board_size * game.board_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        for block in self.resblocks:
            x = block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def predict(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        policy, value = self(state)
        return F.softmax(policy, dim=1).detach().numpy()[0], value.item()

class DynamicNeuralNetwork(NeuralNetwork):
    def __init__(self, game, num_resblocks=19, num_hidden=256):
        super(DynamicNeuralNetwork, self).__init__(game, num_resblocks, num_hidden)
        self.attention = nn.MultiheadAttention(num_hidden, 8)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        for block in self.resblocks:
            x = block(x)
        
        # Apply attention mechanism
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(2, 0, 1)  # (h*w, b, c)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0).view(b, c, h, w)

        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

def train_network(network, optimizer, game_data):
    network.train()
    for state, policy_target, value_target in game_data:
        state = torch.FloatTensor(state).unsqueeze(0)
        policy_target = torch.FloatTensor(policy_target)
        value_target = torch.FloatTensor([value_target])

        optimizer.zero_grad()
        policy, value = network(state)
        policy_loss = F.cross_entropy(policy, policy_target)
        value_loss = F.mse_loss(value, value_target)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    # Example usage
    from src.games.chess import Chess

    game = Chess()
    network = DynamicNeuralNetwork(game)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    # Simulated game data
    game_data = [(game.get_initial_state(), [1/game.action_size]*game.action_size, 0) for _ in range(10)]

    train_network(network, optimizer, game_data)
    print("Network training complete!")
