
import numpy as np
from src.alphazero.neural_network import NeuralNetwork
from src.alphazero.mcts import MCTS
from src.games.chess import Chess
from src.games.go import Go
from src.games.shogi import Shogi

def train_model(game, num_iterations=1000):
    nn = NeuralNetwork(input_shape=game.state_shape, output_shape=game.action_shape)
    mcts = MCTS(nn)
    
    for _ in range(num_iterations):
        state = game.reset()
        while not game.is_terminal(state):
            action = mcts.search(state)
            state = game.step(state, action)
        
        # Update neural network weights based on game outcome
        nn.update(mcts.game_memory)
    
    return nn

def main():
    games = [Chess(), Go(), Shogi()]
    models = {}
    
    for game in games:
        print(f"Training model for {game.__class__.__name__}")
        models[game.__class__.__name__] = train_model(game)
    
    # Save models
    for name, model in models.items():
        model.save(f"models/{name}_model.h5")
    
    print("Pre-trained models have been saved.")

if __name__ == "__main__":
    main()
