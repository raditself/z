
import torch
import torch.nn as nn
import torch.optim as optim

class TransferLearning:
    def __init__(self, source_model, target_model):
        self.source_model = source_model
        self.target_model = target_model

    def transfer_knowledge(self, num_layers_to_freeze=5):
        # Transfer weights from source model to target model
        target_dict = self.target_model.state_dict()
        source_dict = self.source_model.state_dict()
        
        for name, param in source_dict.items():
            if name in target_dict:
                target_dict[name].data.copy_(param.data)

        self.target_model.load_state_dict(target_dict)

        # Freeze the first num_layers_to_freeze layers
        for i, (name, param) in enumerate(self.target_model.named_parameters()):
            if i < num_layers_to_freeze:
                param.requires_grad = False

    def fine_tune(self, target_data, num_epochs=10, learning_rate=0.001):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.target_model.parameters()), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            total_loss = 0
            for state, policy, value in target_data:
                optimizer.zero_grad()
                policy_pred, value_pred = self.target_model(state)
                loss = criterion(policy_pred, policy) + criterion(value_pred, value)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(target_data):.4f}")

def prepare_related_games():
    from src.games.chess import Chess
    from src.games.shogi import Shogi
    from src.alphazero.neural_network import DynamicNeuralNetwork

    chess = Chess()
    shogi = Shogi()

    chess_model = DynamicNeuralNetwork(chess)
    shogi_model = DynamicNeuralNetwork(shogi)

    return chess_model, shogi_model

if __name__ == "__main__":
    source_model, target_model = prepare_related_games()
    
    transfer_learning = TransferLearning(source_model, target_model)
    transfer_learning.transfer_knowledge()

    # Simulate target game data
    target_data = [(torch.randn(1, 3, 8, 8), torch.randn(1, 64*64), torch.randn(1)) for _ in range(100)]

    transfer_learning.fine_tune(target_data)

    print("Transfer learning complete!")
