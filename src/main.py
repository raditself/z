from alphazero.mcts import MCTS
from alphazero.neural_network import NeuralNetwork
from games.chess import Chess

def main():
    game = Chess()
    nn = NeuralNetwork()
    mcts = MCTS()
    
    # Main AlphaZero loop
    while not game.is_game_over():
        state = game.board
        action = mcts.search(state)
        game.make_move(action)
        # Train neural network...

if __name__ == "__main__":
    main()
