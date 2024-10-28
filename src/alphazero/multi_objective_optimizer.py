
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MultiObjectiveNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_objectives):
        super(MultiObjectiveNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_objectives)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MultiObjectiveOptimizer:
    def __init__(self, game, num_objectives, learning_rate=0.001):
        self.game = game
        self.num_objectives = num_objectives
        self.input_size = game.state_size
        self.hidden_size = 128
        self.model = MultiObjectiveNetwork(self.input_size, self.hidden_size, num_objectives)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def update_model(self, state, objectives):
        """Update the multi-objective model based on observed state and objectives."""
        state_tensor = torch.FloatTensor(self.game.state_to_array(state)).unsqueeze(0)
        objectives_tensor = torch.FloatTensor(objectives).unsqueeze(0)
        
        self.optimizer.zero_grad()
        predicted_objectives = self.model(state_tensor)
        loss = nn.MSELoss()(predicted_objectives, objectives_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_objectives(self, state):
        """Predict the objectives given the current state."""
        state_tensor = torch.FloatTensor(self.game.state_to_array(state)).unsqueeze(0)
        with torch.no_grad():
            objectives = self.model(state_tensor).squeeze(0).numpy()
        return objectives
    
    def get_balanced_action(self, state, mcts, weights):
        """Get the action that balances multiple objectives based on given weights."""
        valid_actions = self.game.get_valid_actions(state)
        best_score = float('-inf')
        best_action = None
        
        for action in valid_actions:
            next_state = self.game.get_next_state(state, action)
            objectives = self.predict_objectives(next_state)
            weighted_sum = np.dot(objectives, weights)
            
            # Combine with MCTS score
            mcts_score = mcts.search(next_state)
            combined_score = 0.5 * weighted_sum + 0.5 * mcts_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_action = action
        
        return best_action

def simulate_multi_objective_game(game, mcts, multi_obj_optimizer, num_moves=100):
    state = game.get_initial_state()
    weights = np.array([0.5, 0.3, 0.2])  # Example weights for different objectives
    
    for _ in range(num_moves):
        if game.is_terminal(state):
            break
        
        # Get balanced action
        action = multi_obj_optimizer.get_balanced_action(state, mcts, weights)
        next_state = game.get_next_state(state, action)
        
        # Simulate objectives (in a real scenario, these would be calculated based on the game state)
        objectives = np.random.rand(multi_obj_optimizer.num_objectives)
        
        # Update multi-objective model
        loss = multi_obj_optimizer.update_model(state, objectives)
        print(f"Multi-objective model loss: {loss:.4f}")
        
        state = next_state
    
    return game.get_reward(state)

if __name__ == "__main__":
    from src.games.chess import Chess
    from src.alphazero.mcts import AdaptiveMCTS
    from src.alphazero.neural_network import DynamicNeuralNetwork
    
    game = Chess()
    network = DynamicNeuralNetwork(game)
    mcts = AdaptiveMCTS(game, network)
    multi_obj_optimizer = MultiObjectiveOptimizer(game, num_objectives=3)
    
    # Simulate a game with multi-objective optimization
    reward = simulate_multi_objective_game(game, mcts, multi_obj_optimizer)
    print(f"Game result: {reward}")
