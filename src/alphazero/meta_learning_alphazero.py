
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import logging
from .alphazero import AlphaZero

class MetaLearningAlphaZero(AlphaZero):
    def __init__(self, game, args):
        super().__init__(game, args)
        self.base_models = OrderedDict()
        self.meta_lr = args.get('meta_lr', 0.01)
        self.current_iteration = 0
        self.style_history = []
        self.logger = logging.getLogger(__name__)

    def add_base_model(self, game_type, model):
        self.base_models[game_type] = model
        self.logger.info(f"Added base model for game type: {game_type}")

    def extract_game_features(self, game):
        features = [
            game.board_size,
            game.action_size,
            game.num_players,
            int(game.is_zero_sum),
            int(game.is_perfect_information)
        ]
        return torch.tensor(features, dtype=torch.float32)

    def select_base_model(self, game):
        game_features = self.extract_game_features(game)
        best_similarity = float('-inf')
        best_model = None
        selected_game_type = None

        for game_type, model in self.base_models.items():
            similarity = self.compute_similarity(game_features, game_type)
            if similarity > best_similarity:
                best_similarity = similarity
                best_model = model
                selected_game_type = game_type

        if best_model is None:
            self.logger.warning("No suitable base model found. Using the default model.")
        else:
            self.logger.info(f"Selected base model for game type: {selected_game_type}")

        return best_model, selected_game_type

    def compute_similarity(self, game_features, game_type):
        base_features = self.extract_game_features(self.base_models[game_type].game)
        return 1 / (1 + torch.norm(game_features - base_features))

    def adapt_to_new_game(self, game, num_iterations=5):
        base_model, selected_game_type = self.select_base_model(game)
        if base_model is None:
            return

        self.net.load_state_dict(base_model.state_dict())

        for i in range(num_iterations):
            examples = self.self_play(1)  # Play one game
            self.meta_update(examples)
            self.logger.info(f"Completed adaptation iteration {i+1}/{num_iterations}")

        # Update the base model after adaptation
        self.update_base_model(selected_game_type)

    def meta_update(self, examples):
        self.net.train()
        
        # Compute meta-gradient
        meta_grad = OrderedDict()
        for name, param in self.net.named_parameters():
            meta_grad[name] = torch.zeros_like(param)

        for example in examples:
            state, policy_target, value_target, game_phase = example
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_target = torch.FloatTensor(policy_target).unsqueeze(0).to(self.device)
            value_target = torch.FloatTensor([value_target]).to(self.device)
            game_phase = torch.FloatTensor([game_phase]).to(self.device)

            # Forward pass
            out_policy, out_value = self.net(state, game_phase)
            loss = self.compute_loss(out_policy, out_value, policy_target, value_target)

            # Compute gradients
            grads = torch.autograd.grad(loss, self.net.parameters())

            # Accumulate meta-gradients
            for name, param in self.net.named_parameters():
                meta_grad[name] += grads[list(self.net.named_parameters()).index((name, param))]

        # Update model parameters
        for name, param in self.net.named_parameters():
            param.data -= self.meta_lr * meta_grad[name]

    def compute_loss(self, out_policy, out_value, policy_target, value_target):
        policy_loss = nn.CrossEntropyLoss()(out_policy, policy_target)
        value_loss = nn.MSELoss()(out_value.squeeze(-1), value_target)
        return policy_loss + value_loss

    def load_checkpoint(self, filepath):
        try:
            checkpoint = torch.load(filepath)
            self.current_iteration = checkpoint['iteration']
            self.net.load_state_dict(checkpoint['state_dict'])
            self.base_models = checkpoint['base_models']
            self.style_history = checkpoint.get('style_history', [])
            self.logger.info(f"Loaded checkpoint from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")

    def save_checkpoint(self, filepath):
        try:
            torch.save({
                'iteration': self.current_iteration,
                'state_dict': self.net.state_dict(),
                'base_models': self.base_models,
                'style_history': self.style_history
            }, filepath)
            self.logger.info(f"Saved checkpoint to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

    def update_base_model(self, game_type):
        if game_type in self.base_models:
            self.base_models[game_type].load_state_dict(self.net.state_dict())
            self.logger.info(f"Updated base model for game type: {game_type}")
        else:
            self.logger.warning(f"No base model found for game type: {game_type}")


    def update_style_history(self, style_data):
        self.style_history.append(style_data)

