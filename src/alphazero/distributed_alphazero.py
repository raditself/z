
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from .alphazero import AlphaZero
from .logger import AlphaZeroLogger
from .evaluator import AlphaZeroEvaluator, load_model
from .alphazero_net import AlphaZeroNet

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class DistributedAlphaZero(AlphaZero):
    def __init__(self, game, args, rank, world_size):
        super().__init__(game, args)
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.net = self.net.to(self.device)
        self.net = DDP(self.net, device_ids=[rank])
        self.logger = AlphaZeroLogger(os.path.join(args.log_dir, f'worker_{rank}'))
        self.evaluator = AlphaZeroEvaluator(game, args)

    def distributed_self_play(self):
        examples = self.self_play()
        gathered_examples = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_examples, examples)
        all_examples = [ex for worker_examples in gathered_examples for ex in worker_examples]
        self.logger.log_info(f"Worker {self.rank}: Generated {len(examples)} examples")
        return all_examples

    def distributed_learn(self):
        best_model_path = None
        for iteration in range(self.args.num_iterations):
            self.logger.log_info(f"Worker {self.rank}: Starting iteration {iteration}")
            examples = self.distributed_self_play()
            
            policy_loss, value_loss = self.train(examples)
            dist.all_reduce(policy_loss)
            dist.all_reduce(value_loss)
            policy_loss /= self.world_size
            value_loss /= self.world_size
            
            if self.rank == 0:
                self.logger.log_info(f"Training on {len(examples)} examples")
                self.logger.log_scalar('policy_loss', policy_loss.item(), iteration)
                self.logger.log_scalar('value_loss', value_loss.item(), iteration)
                
                current_model_path = f'model_iter_{iteration}.pth'
                torch.save(self.net.module.state_dict(), current_model_path)
                
                if best_model_path:
                    best_model = load_model(AlphaZeroNet, best_model_path, self.game, self.args)
                    current_model = load_model(AlphaZeroNet, current_model_path, self.game, self.args)
                    
                    wins = self.evaluator.evaluate(current_model, best_model)
                    win_rate = wins[1] / sum(wins.values())
                    self.logger.log_info(f"Evaluation results: {wins}")
                    self.logger.log_scalar('win_rate', win_rate, iteration)
                    
                    if win_rate > 0.55:  # If new model wins more than 55% of games
                        best_model_path = current_model_path
                        self.logger.log_info(f"New best model: {best_model_path}")
                else:
                    best_model_path = current_model_path
            
            dist.barrier()
            for name, param in self.net.named_parameters():
                dist.broadcast(param.data, 0)
                if self.rank == 0:
                    self.logger.log_histogram(f'param_{name}', param.data, iteration)

            self.logger.log_info(f"Worker {self.rank}: Finished iteration {iteration}")

    def train(self, examples):
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        for _ in range(self.args.epochs):
            for batch in self.get_batch(examples):
                state, policy_targets, value_targets = batch
                state = state.to(self.device)
                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device)

                out_policy, out_value = self.net(state)

                policy_loss = self.loss_policy(out_policy, policy_targets)
                value_loss = self.loss_value(out_value.squeeze(-1), value_targets)
                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

        return total_policy_loss / num_batches, total_value_loss / num_batches

def run_worker(rank, world_size, game, args):
    setup(rank, world_size)
    alphazero = DistributedAlphaZero(game, args, rank, world_size)
    alphazero.distributed_learn()
    alphazero.logger.close()
    cleanup()

def distributed_main(game, args, num_workers):
    mp.spawn(run_worker, args=(num_workers, game, args), nprocs=num_workers, join=True)

if __name__ == "__main__":
    from .tictactoe import TicTacToe
    import argparse

    parser = argparse.ArgumentParser(description='Train Distributed AlphaZero for Tic-Tac-Toe')
    # Add all necessary arguments here
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for storing logs')
    args = parser.parse_args()

    game = TicTacToe()
    num_workers = 4  # Or any other number of workers you want to use
    distributed_main(game, args, num_workers)
