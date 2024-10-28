
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class OpponentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OpponentModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

class AdaptiveOpponentModeling:
    def __init__(self, game, learning_rate=0.001):
        self.game = game
        self.input_size = game.state_size
        self.output_size = game.action_size
        self.hidden_size = 128
        self.model = OpponentModel(self.input_size, self.hidden_size, self.output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def update_model(self, state, action):
        """Update the opponent model based on observed state-action pair."""
        state_tensor = torch.FloatTensor(self.game.state_to_array(state)).unsqueeze(0)
        action_tensor = torch.LongTensor([action])
        
        self.optimizer.zero_grad()
        predicted_action_probs = self.model(state_tensor)
        loss = self.criterion(predicted_action_probs, action_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_opponent_action(self, state):
        """Predict the opponent's next action given the current state."""
        state_tensor = torch.FloatTensor(self.game.state_to_array(state)).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.model(state_tensor).squeeze(0).numpy()
        return action_probs
    
    def get_best_response(self, state, mcts):
        """Get the best response action against the predicted opponent action."""
        opponent_action_probs = self.predict_opponent_action(state)
        
        best_response_value = float('-inf')
        best_response_action = None
        
        for action in self.game.get_valid_actions(state):
            next_state = self.game.get_next_state(state, action)
            opponent_action = np.argmax(opponent_action_probs)
            final_state = self.game.get_next_state(next_state, opponent_action)
            
            # Use MCTS to evaluate the resulting position
            mcts_value = mcts.search(final_state)
            
            if mcts_value > best_response_value:
                best_response_value = mcts_value
                best_response_action = action
        
        return best_response_action

def simulate_game_with_opponent_modeling(game, mcts, opponent_model, num_moves=100):
    state = game.get_initial_state()
    for _ in range(num_moves):
        if game.is_terminal(state):
            break
        
        # AlphaZero's turn
        alphazero_action = mcts.search(state)
        state = game.get_next_state(state, alphazero_action)
        
        if game.is_terminal(state):
            break
        
        # Opponent's turn (simulated)
        opponent_action_probs = opponent_model.predict_opponent_action(state)
        opponent_action = np.random.choice(game.action_size, p=opponent_action_probs)
        state = game.get_next_state(state, opponent_action)
        
        # Update opponent model
        loss = opponent_model.update_model(state, opponent_action)
        print(f"Opponent model loss: {loss:.4f}")
    
    return game.get_reward(state)

if __name__ == "__main__":
    from src.games.chess import Chess
    from src.alphazero.mcts import AdaptiveMCTS
    from src.alphazero.neural_network import DynamicNeuralNetwork
    
    game = Chess()
    network = DynamicNeuralNetwork(game)
    mcts = AdaptiveMCTS(game, network)
    opponent_model = AdaptiveOpponentModeling(game)
    
    # Simulate a game with opponent modeling
    reward = simulate_game_with_opponent_modeling(game, mcts, opponent_model)
    print(f"Game result: {reward}")
