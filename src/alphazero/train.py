import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from chess_dataset import ChessDataset
from model import get_model
from neural_architecture_search import neural_architecture_search
import time

def train(config):
    # Run neural architecture search
    print("Running neural architecture search...")
    best_model, best_performance = neural_architecture_search(config.db_path, num_trials=config.nas_trials)
    print(f"Best model performance: {best_performance}")
    print(f"Best model architecture: {best_model}")

    # Save the best model
    torch.save(best_model.state_dict(), "best_model.pth")

    # Load the best model
    model = get_model("best_model.pth")

    # Set up data loaders
    print("Loading dataset...")
    start_time = time.time()
    full_dataset = ChessDataset(config.db_path)
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

    # Use only a subset of the data for testing
    subset_size = min(10000, len(full_dataset))
    subset_indices = torch.randperm(len(full_dataset))[:subset_size]
    dataset = Subset(full_dataset, subset_indices)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config.num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {total_loss / (batch_idx + 1):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Time: {epoch_time:.2f} seconds")

    # Save the final model
    torch.save(model.state_dict(), "final_model.pth")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the chess model")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the SQLite database")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--nas_trials", type=int, default=5, help="Number of trials for neural architecture search")

    config = parser.parse_args()
    train(config)

