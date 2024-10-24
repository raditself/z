
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.alphazero.self_play import execute_self_play
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainAlphaZero:
    def __init__(self, game, model, args, load_path=None):
        self.game = game
        self.model = model
        self.args = args
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.elo_rating = 1500  # Initial Elo rating
        self.start_iteration = 0

        if load_path:
            self.load_model(load_path)

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.elo_rating = checkpoint['elo_rating']
        self.start_iteration = checkpoint['iteration']
        logger.info(f"Loaded model from {load_path}")
        logger.info(f"Resuming from iteration {self.start_iteration}")

    def save_model(self, path, iteration):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'elo_rating': self.elo_rating,
            'iteration': iteration
        }, path)
        logger.info(f'Model saved at {path}')

    def calculate_rates(self, results):
        total_games = len(results)
        wins = sum(1 for r in results if r == 1)
        draws = sum(1 for r in results if r == 0)
        losses = sum(1 for r in results if r == -1)
        
        return {
            'win_rate': wins / total_games,
            'draw_rate': draws / total_games,
            'loss_rate': losses / total_games
        }

    def update_elo_rating(self, opponent_rating, score, k=32):
        expected_score = 1 / (1 + 10 ** ((opponent_rating - self.elo_rating) / 400))
        self.elo_rating += k * (score - expected_score)

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

    def train_iteration(self):
        examples = []
        results = []
        for _ in range(self.args.episodes):
            game_examples, result = execute_self_play(self.game, self.model, self.args.mcts_simulations)
            examples.extend(game_examples)
            results.append(result)
        
        self.train(examples)
        
        # Calculate metrics
        rates = self.calculate_rates(results)
        
        # Update Elo rating (assuming playing against a fixed opponent)
        opponent_rating = 1500  # Fixed opponent rating for simplicity
        average_score = sum(results) / len(results)
        self.update_elo_rating(opponent_rating, average_score)
        
        self.start_iteration += 1
        
        if self.start_iteration % 10 == 0:
            self.save_model(f'alphazero_model_iter_{self.start_iteration}.pth', self.start_iteration)
        
        return self.start_iteration, rates['win_rate'], rates['loss_rate'], rates['draw_rate']

    def learn(self):
        while self.start_iteration < self.args.iterations:
            iteration, win_rate, loss_rate, draw_rate = self.train_iteration()
            
            logger.info(f"Iteration {iteration}:")
            logger.info(f"Win rate: {win_rate:.2f}")
            logger.info(f"Draw rate: {draw_rate:.2f}")
            logger.info(f"Loss rate: {loss_rate:.2f}")
            logger.info(f"Current Elo rating: {self.elo_rating:.2f}")

# Remove the main function as it's not needed anymore
