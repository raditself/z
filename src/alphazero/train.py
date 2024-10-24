
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.alphazero.self_play import execute_self_play

class TrainAlphaZero:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)  # Default learning rate

    def train(self, examples):
        self.model.train()
        
        # Prepare data
        states, pis, vs = zip(*examples)
        states = torch.FloatTensor(states)
        pis = torch.FloatTensor(pis)
        vs = torch.FloatTensor(vs)
        
        dataset = TensorDataset(states, pis, vs)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Default batch size

        for epoch in range(10):  # Default number of epochs
            for batch_idx, (states, pis, vs) in enumerate(dataloader):
                out_pis, out_vs = self.model(states)
                
                pi_loss = F.cross_entropy(out_pis, pis)
                v_loss = F.mse_loss(out_vs.squeeze(), vs)
                total_loss = pi_loss + v_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    def learn(self):
        for i in range(self.args.iterations):
            examples = []
            for _ in range(self.args.episodes):
                examples.extend(execute_self_play(self.game, self.model, self.args.mcts_simulations))
            
            self.train(examples)
            
            # Here you might want to add code to evaluate the model and save it if it's better
            if (i + 1) % 10 == 0:
                torch.save(self.model.state_dict(), f'alphazero_model_iter_{i+1}.pth')
                print(f'Model saved at iteration {i+1}')

# Remove the main function as it's not needed anymore
