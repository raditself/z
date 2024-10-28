
# Updated content for nnet.py
import os
import torch
import torch.optim as optim
from alphazero_net import AlphaZeroNet, ResNetAlphaZero, TransformerAlphaZero

class NNetWrapper:
    def __init__(self, game, architecture):
        self.game = game
        self.action_size = game.getActionSize()
        self.architecture = architecture
        
        if architecture == AlphaZeroNet:
            self.nnet = AlphaZeroNet(game, self.action_size)
        elif architecture == ResNetAlphaZero:
            self.nnet = ResNetAlphaZero(game.board_size, self.action_size)
        elif architecture == TransformerAlphaZero:
            self.nnet = TransformerAlphaZero(game.board_size, self.action_size)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        self.optimizer = optim.Adam(self.nnet.parameters())

    def train(self, examples):
        # ... (keep existing training code)
        pass

    def predict(self, board):
        # ... (keep existing prediction code)
        pass

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'architecture': self.architecture.__name__
        }, filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        checkpoint = torch.load(filepath)
        
        architecture_name = checkpoint['architecture']
        if architecture_name == 'AlphaZeroNet':
            self.nnet = AlphaZeroNet(self.game, self.action_size)
        elif architecture_name == 'ResNetAlphaZero':
            self.nnet = ResNetAlphaZero(self.game.board_size, self.action_size)
        elif architecture_name == 'TransformerAlphaZero':
            self.nnet = TransformerAlphaZero(self.game.board_size, self.action_size)
        else:
            raise ValueError(f"Unknown architecture: {architecture_name}")
        
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer = optim.Adam(self.nnet.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer'])
