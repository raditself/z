
import numpy as np
from tensorflow import keras
from keras_tuner import RandomSearch
from src.alphazero.model import ChessModel, CheckersModel

def build_model(hp, game_type='chess'):
    model = keras.Sequential()
    
    # Input layer
    if game_type == 'chess':
        input_shape = (8, 8, 12)  # 8x8 board, 12 piece types
    elif game_type == 'checkers':
        input_shape = (8, 8, 4)   # 8x8 board, 4 piece types (regular and king for each color)
    
    model.add(keras.layers.Input(shape=input_shape))
    
    # Convolutional layers
    for i in range(hp.Int('num_conv_layers', 2, 5)):
        model.add(keras.layers.Conv2D(
            filters=hp.Int(f'conv_{i}_filters', 32, 256, step=32),
            kernel_size=hp.Choice(f'conv_{i}_kernel', values=[3, 5]),
            activation='relu',
            padding='same'
        ))
        
        if hp.Boolean(f'batch_norm_{i}'):
            model.add(keras.layers.BatchNormalization())
    
    # Flatten the output
    model.add(keras.layers.Flatten())
    
    # Dense layers
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'dense_{i}_units', 64, 512, step=64),
            activation='relu'
        ))
        
        if hp.Boolean(f'dropout_{i}'):
            model.add(keras.layers.Dropout(hp.Float(f'dropout_{i}_rate', 0.1, 0.5, step=0.1)))
    
    # Output layers
    if game_type == 'chess':
        model.add(keras.layers.Dense(64, activation='softmax', name='policy_output'))  # 64 possible moves in chess
    elif game_type == 'checkers':
        model.add(keras.layers.Dense(32, activation='softmax', name='policy_output'))  # 32 possible moves in checkers
    
    model.add(keras.layers.Dense(1, activation='tanh', name='value_output'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mean_squared_error'},
        loss_weights={'policy_output': 1.0, 'value_output': 1.0},
        metrics={'policy_output': 'accuracy', 'value_output': 'mae'}
    )
    
    return model

def neural_architecture_search(game_type='chess', max_trials=50, epochs=10):
    if game_type == 'chess':
        base_model = ChessModel()
    elif game_type == 'checkers':
        base_model = CheckersModel()
    else:
        raise ValueError("Invalid game type. Choose 'chess' or 'checkers'.")
    
    tuner = RandomSearch(
        lambda hp: build_model(hp, game_type),
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='nas_results',
        project_name=f'{game_type}_nas'
    )
    
    # Generate some dummy data for training
    x_train = np.random.rand(1000, *base_model.input_shape)
    y_train = {
        'policy_output': np.random.rand(1000, base_model.policy_dim),
        'value_output': np.random.rand(1000, 1)
    }
    
    tuner.search(x_train, y_train, epochs=epochs, validation_split=0.2)
    
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("Best hyperparameters:")
    print(best_hp.values)
    
    return best_model, best_hp

# Usage example:
# best_chess_model, best_chess_hp = neural_architecture_search('chess')
# best_checkers_model, best_checkers_hp = neural_architecture_search('checkers')
