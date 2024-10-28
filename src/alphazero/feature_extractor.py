
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class FeatureExtractor:
    def __init__(self, input_dim, hidden_dim=64, learning_rate=0.001, batch_size=32, num_epochs=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.autoencoder = AutoEncoder(input_dim, hidden_dim)
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train(self, raw_states):
        dataset = TensorDataset(torch.FloatTensor(raw_states))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in dataloader:
                inputs = batch[0]
                self.optimizer.zero_grad()
                _, reconstructed = self.autoencoder(inputs)
                loss = self.criterion(reconstructed, inputs)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

    def extract_features(self, raw_state):
        with torch.no_grad():
            features, _ = self.autoencoder(torch.FloatTensor(raw_state).unsqueeze(0))
        return features.squeeze(0).numpy()

def prepare_raw_states(game, num_states=10000):
    raw_states = []
    for _ in range(num_states):
        state = game.get_random_state()
        raw_state = game.state_to_array(state)
        raw_states.append(raw_state)
    return raw_states

if __name__ == "__main__":
    from src.games.chess import Chess

    game = Chess()
    raw_states = prepare_raw_states(game)
    input_dim = raw_states[0].size

    feature_extractor = FeatureExtractor(input_dim)
    feature_extractor.train(raw_states)

    # Test feature extraction
    test_state = game.get_random_state()
    test_raw_state = game.state_to_array(test_state)
    extracted_features = feature_extractor.extract_features(test_raw_state)
    print("Extracted features shape:", extracted_features.shape)
