
from .game import Game
from .model import AlphaZeroNetwork, export_model, import_model
from .self_play import execute_self_play
from .logging_system import AdvancedLogger
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_network(game, model, args, logger):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for iteration in range(args.num_iterations):
        # Self-play
        examples = execute_self_play(model, game, args.num_self_play_games, args)
        
        # Train the model
        model.train()
        for epoch in range(args.epochs):
            batch_idx = 0
            while batch_idx < len(examples):
                sample = examples[batch_idx:min(len(examples), batch_idx + args.batch_size)]
                boards, pis, vs = zip(*sample)
                
                boards = torch.FloatTensor(np.array(boards))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # Compute output and loss
                out_pi, out_v = model(boards)
                pi_loss = F.cross_entropy(out_pi, target_pis)
                v_loss = F.mse_loss(out_v.squeeze(), target_vs)
                total_loss = pi_loss + v_loss

                # Backward and optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += args.batch_size

        # Evaluate the model
        model.eval()
        test_examples = execute_self_play(model, game, args.num_test_games, args)
        test_boards, test_pis, test_vs = zip(*test_examples)
        test_boards = torch.FloatTensor(np.array(test_boards))
        test_pis = torch.FloatTensor(np.array(test_pis))
        test_vs = torch.FloatTensor(np.array(test_vs).astype(np.float64))

        with torch.no_grad():
            test_out_pi, test_out_v = model(test_boards)
            test_pi_loss = F.cross_entropy(test_out_pi, test_pis)
            test_v_loss = F.mse_loss(test_out_v.squeeze(), test_vs)
            test_total_loss = test_pi_loss + test_v_loss

        # Log the results
        logger.log_training_progress(iteration, test_total_loss.item(), 0, 0)  # TODO: Add accuracy and Elo rating

        # Save the model
        if iteration % args.checkpoint_interval == 0:
            export_model(model, f'model_iteration_{iteration}.pth')

    return model

if __name__ == '__main__':
    game = Game(variant='standard')  # You can change this to 'mini' or 'grand' for different variants
    input_shape = (3, game.board_size, game.board_size)  # Adjust based on the game representation
    model = AlphaZeroNetwork(input_shape, game.action_size)
    
    class Args:
        num_iterations = 100
        num_self_play_games = 100
        num_test_games = 20
        epochs = 10
        batch_size = 64
        learning_rate = 0.001
        checkpoint_interval = 10
        dirichlet_alpha = 0.3
        exploration_fraction = 0.25

    args = Args()
    logger = AdvancedLogger()

    trained_model = train_network(game, model, args, logger)
    export_model(trained_model, 'final_model.pth')
