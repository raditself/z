
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class HumanKnowledgeIntegration:
    def __init__(self, game, base_model):
        self.game = game
        self.base_model = base_model
        self.human_policy_net = nn.Linear(game.action_size, game.action_size)
        self.optimizer = optim.Adam(self.human_policy_net.parameters(), lr=0.001)

    def integrate_human_knowledge(self, human_moves, num_epochs=100):
        """
        Integrate human knowledge into the model.
        human_moves: list of tuples (state, action) representing expert human moves
        """
        for epoch in range(num_epochs):
            total_loss = 0
            for state, action in human_moves:
                state_tensor = torch.FloatTensor(self.game.state_to_array(state)).unsqueeze(0)
                action_tensor = torch.LongTensor([action])

                # Get base model's policy
                with torch.no_grad():
                    base_policy, _ = self.base_model(state_tensor)

                # Apply human knowledge adjustment
                adjusted_policy = self.human_policy_net(base_policy)
                
                # Compute loss
                loss = nn.CrossEntropyLoss()(adjusted_policy, action_tensor)
                
                # Backpropagate and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(human_moves):.4f}')

    def get_integrated_policy(self, state):
        """
        Get the policy that integrates both the base model and human knowledge.
        """
        state_tensor = torch.FloatTensor(self.game.state_to_array(state)).unsqueeze(0)
        with torch.no_grad():
            base_policy, value = self.base_model(state_tensor)
            integrated_policy = self.human_policy_net(base_policy)
        return integrated_policy.squeeze(0).numpy(), value.item()

def collect_human_moves(game, num_moves=1000):
    """
    Simulate collecting expert human moves.
    In a real scenario, this would involve actual human experts playing the game.
    """
    human_moves = []
    for _ in range(num_moves):
        state = game.get_random_state()
        action = np.random.choice(game.get_valid_actions(state))  # Simulated expert move
        human_moves.append((state, action))
    return human_moves

if __name__ == "__main__":
    from src.games.chess import Chess
    from src.alphazero.neural_network import DynamicNeuralNetwork

    game = Chess()
    base_model = DynamicNeuralNetwork(game)
    
    human_knowledge_integrator = HumanKnowledgeIntegration(game, base_model)
    
    # Collect simulated human expert moves
    human_moves = collect_human_moves(game)
    
    # Integrate human knowledge
    human_knowledge_integrator.integrate_human_knowledge(human_moves)
    
    # Test integrated policy
    test_state = game.get_random_state()
    integrated_policy, value = human_knowledge_integrator.get_integrated_policy(test_state)
    print("Integrated policy:", integrated_policy)
    print("Value:", value)
