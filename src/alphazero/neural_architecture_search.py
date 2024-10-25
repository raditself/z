import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from chess_dataset import ChessDataset
import time

def create_model(num_layers, neurons_per_layer, activation_func, dropout_rate):
    layers = []
    input_size = 64 * 12  # 8x8 board, 12 piece types (6 white, 6 black)
    for i in range(num_layers):
        layers.append(nn.Linear(input_size if i == 0 else neurons_per_layer, neurons_per_layer))
        layers.append(activation_func())
        layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(neurons_per_layer, 1))  # Output layer for evaluation
    return nn.Sequential(*layers)

def train_and_evaluate(model, train_loader, val_loader, epochs=1):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def neural_architecture_search(db_path, num_trials=5):
    print("Loading dataset...")
    start_time = time.time()
    full_dataset = ChessDataset(db_path)
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

    # Use only a small subset of the data for testing
    subset_size = min(1000, len(full_dataset))
    subset_indices = torch.randperm(len(full_dataset))[:subset_size]
    dataset = Subset(full_dataset, subset_indices)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    best_model = None
    best_performance = float('inf')

    for trial in range(num_trials):
        print(f"Starting trial {trial + 1}/{num_trials}")
        start_time = time.time()

        num_layers = random.randint(2, 3)
        neurons_per_layer = random.choice([64, 128])
        activation_func = random.choice([nn.ReLU, nn.LeakyReLU])
        dropout_rate = random.uniform(0, 0.3)

        model = create_model(num_layers, neurons_per_layer, activation_func, dropout_rate)
        performance = train_and_evaluate(model, train_loader, val_loader)

        print(f"Trial {trial + 1}/{num_trials}: Performance = {performance:.4f}, Time taken: {time.time() - start_time:.2f} seconds")

        if performance < best_performance:
            best_performance = performance
            best_model = model

    return best_model, best_performance

if __name__ == "__main__":
    best_model, best_performance = neural_architecture_search("chess_positions.db")
    print(f"Best model performance: {best_performance}")
    print(f"Best model architecture: {best_model}")

    # Save the best model
    torch.save(best_model.state_dict(), "best_model.pth")
