
from .game import Game
from .model import AlphaZeroNetwork, export_model, import_model
from .self_play import execute_self_play
from .logging_system import AdvancedLogger
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
import os
import argparse

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_network(rank, world_size, game, model, args, logger):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    for iteration in range(args.num_iterations):
        # Self-play
        examples = execute_self_play(model.module, game, args.num_self_play_games // world_size, args)
        
        # Synchronize examples across all processes
        all_examples = [None for _ in range(world_size)]
        dist.all_gather_object(all_examples, examples)
        examples = [ex for exs in all_examples for ex in exs]
        
        # Create dataset and sampler
        dataset = TensorDataset(
            torch.FloatTensor(np.array([ex[0] for ex in examples])),
            torch.FloatTensor(np.array([ex[1] for ex in examples])),
            torch.FloatTensor(np.array([ex[2] for ex in examples]).astype(np.float64))
        )
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
        
        # Train the model
        model.train()
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            for boards, target_pis, target_vs in dataloader:
                boards, target_pis, target_vs = boards.to(device), target_pis.to(device), target_vs.to(device)

                # Compute output and loss
                out_pi, out_v = model(boards)
                pi_loss = F.cross_entropy(out_pi, target_pis)
                v_loss = F.mse_loss(out_v.squeeze(), target_vs)
                total_loss = pi_loss + v_loss

                # Backward and optimize
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        scheduler.step()

        # Evaluate the model (only on rank 0)
        if rank == 0:
            model.eval()
            test_examples = execute_self_play(model.module, game, args.num_test_games, args)
            test_boards, test_pis, test_vs = zip(*test_examples)
            test_boards = torch.FloatTensor(np.array(test_boards)).to(device)
            test_pis = torch.FloatTensor(np.array(test_pis)).to(device)
            test_vs = torch.FloatTensor(np.array(test_vs).astype(np.float64)).to(device)

            with torch.no_grad():
                test_out_pi, test_out_v = model(test_boards)
                test_pi_loss = F.cross_entropy(test_out_pi, test_pis)
                test_v_loss = F.mse_loss(test_out_v.squeeze(), test_vs)
                test_total_loss = test_pi_loss + test_v_loss

            # Log the results
            logger.log_training_progress(iteration, test_total_loss.item(), optimizer.param_groups[0]['lr'], 0)  # TODO: Add accuracy and Elo rating

            # Save the model
            if iteration % args.checkpoint_interval == 0:
                export_model(model.module, f'model_iteration_{iteration}.pth')

    cleanup()
    return model.module if rank == 0 else None

def main(rank, world_size, args):
    game = Game(variant='standard')  # You can change this to 'mini' or 'grand' for different variants
    input_shape = (3, game.board_size, game.board_size)  # Adjust based on the game representation
    model = AlphaZeroNetwork(input_shape, game.action_size)
    
    logger = AdvancedLogger()

    trained_model = train_network(rank, world_size, game, model, args, logger)
    if rank == 0:
        export_model(trained_model, 'final_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed AlphaZero Training')
    parser.add_argument('--world_size', type=int, default=1, help='number of processes for distributed training')
    parser.add_argument('--num_iterations', type=int, default=100, help='number of training iterations')
    parser.add_argument('--num_self_play_games', type=int, default=100, help='number of self-play games per iteration')
    parser.add_argument('--num_test_games', type=int, default=20, help='number of test games per iteration')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs per iteration')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate for optimizer')
    parser.add_argument('--lr_step_size', type=int, default=10, help='step size for learning rate scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='gamma for learning rate scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max norm of the gradients')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='save model every N iterations')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3, help='Dirichlet noise alpha parameter')
    parser.add_argument('--exploration_fraction', type=float, default=0.25, help='fraction of moves for exploration')
    args = parser.parse_args()

    world_size = args.world_size
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
