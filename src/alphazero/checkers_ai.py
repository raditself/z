
import numpy as np
from src.alphazero.mcts_nn import MCTS
from src.alphazero.model import CheckersModel
from src.games.checkers import CheckersGame

class CheckersAI:
    def __init__(self, model_path=None):
        self.game = CheckersGame()
        self.model = CheckersModel()
        if model_path:
            self.model.load(model_path)
        self.mcts = MCTS(self.game, self.model)

    def get_move(self, board, temperature=1.0):
        state = self.game.get_state(board)
        action_probs = self.mcts.search(state)
        
        # Apply temperature
        if temperature == 0:
            action = np.argmax(action_probs)
        else:
            action_probs = np.power(action_probs, 1/temperature)
            action_probs /= np.sum(action_probs)
            action = np.random.choice(len(action_probs), p=action_probs)
        
        return self.game.action_to_move(action)

    def train(self, num_iterations, num_episodes, num_epochs):
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Self-play
            examples = []
            for episode in range(num_episodes):
                board = self.game.get_initial_board()
                episode_steps = 0
                while not self.game.is_game_over(board):
                    state = self.game.get_state(board)
                    action_probs = self.mcts.search(state)
                    examples.append((state, action_probs, None))  # Winner unknown yet
                    
                    action = np.random.choice(len(action_probs), p=action_probs)
                    board = self.game.get_next_state(board, action)
                    episode_steps += 1
                
                # Get game result
                winner = self.game.get_winner(board)
                examples = [(x[0], x[1], winner) for x in examples]
            
            # Train neural network
            self.model.train(examples, num_epochs)
            
            # Save the model
            self.model.save(f"checkers_model_iteration_{iteration}.h5")

# Usage example:
# ai = CheckersAI()
# ai.train(num_iterations=10, num_episodes=100, num_epochs=10)
# best_move = ai.get_move(current_board)
