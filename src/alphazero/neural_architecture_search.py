import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from chess_dataset import ChessDataset
import time

def create_model(num_layers, neurons_per_layer, activation_func, dropout_rate):
    model = models.Sequential()
    input_size = 64 * 12  # 8x8 board, 12 piece types (6 white, 6 black)
    model.add(layers.Input(shape=(input_size,)))
    for _ in range(num_layers):
        model.add(layers.Dense(neurons_per_layer, activation=activation_func))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))  # Output layer for evaluation
    return model

def train_and_evaluate(model, train_dataset, val_dataset, epochs=1):
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=0)
    
    results = model.evaluate(val_dataset, verbose=0)
    return results

def neural_architecture_search(db_path, num_trials=5):
    print("Loading dataset...")
    start_time = time.time()
    full_dataset = ChessDataset(db_path)
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

    # Use only a small subset of the data for testing
    subset_size = min(1000, len(full_dataset))
    full_dataset = full_dataset.shuffle(buffer_size=len(full_dataset)).take(subset_size)

    train_size = int(0.8 * subset_size)
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)

    train_dataset = train_dataset.batch(32)
    val_dataset = val_dataset.batch(32)

    print(f"Train dataset size: {train_size}")
    print(f"Validation dataset size: {subset_size - train_size}")

    best_model = None
    best_performance = float('inf')

    for trial in range(num_trials):
        print(f"Starting trial {trial + 1}/{num_trials}")
        start_time = time.time()

        num_layers = random.randint(2, 3)
        neurons_per_layer = random.choice([64, 128])
        activation_func = random.choice(['relu', 'leaky_relu'])
        dropout_rate = random.uniform(0, 0.3)

        model = create_model(num_layers, neurons_per_layer, activation_func, dropout_rate)
        performance = train_and_evaluate(model, train_dataset, val_dataset)

        print(f"Trial {trial + 1}/{num_trials}: Performance = {performance:.4f}, Time taken: {time.time() - start_time:.2f} seconds")

        if performance < best_performance:
            best_performance = performance
            best_model = model

    return best_model, best_performance

if __name__ == "__main__":
    best_model, best_performance = neural_architecture_search("chess_positions.db")
    print(f"Best model performance: {best_performance}")
    print(f"Best model architecture: {best_model.summary()}")

    # Save the best model
    best_model.save('best_model.h5')
