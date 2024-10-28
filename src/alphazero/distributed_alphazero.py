
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from .alphazero import AlphaZero
from .logger import AlphaZeroLogger
from .evaluator import AlphaZeroEvaluator, load_model
from .alphazero_net import AlphaZeroNet
from .parameter_server import ParameterServer, run_parameter_server

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
        self.logger = AlphaZeroLogger(os.path.join(args.log_dir, f'worker_{rank}'))
        self.evaluator = AlphaZeroEvaluator(game, args)
        
        if self.rank == 0:
            self.parameter_server = ParameterServer(self.net)

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
            
            if self.rank == 0:
                self.logger.log_info(f"Training on {len(examples)} examples")
                self.logger.log_scalar('policy_loss', policy_loss, iteration)
                self.logger.log_scalar('value_loss', value_loss, iteration)
                
                current_model_path = f'model_iter_{iteration}.pth'
                torch.save(self.net.state_dict(), current_model_path)
                
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
            
            # Synchronize model parameters
            self.sync_parameters()

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

                # Calculate gradients
                loss.backward()

                # Send gradients to parameter server
                self.send_gradients()

                # Receive updated parameters from parameter server
                self.sync_parameters()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

        return total_policy_loss / num_batches, total_value_loss / num_batches

    def send_gradients(self):
        gradients = [param.grad for param in self.net.parameters()]
        for grad in gradients:
            dist.send(grad, dst=0)

    def sync_parameters(self):
        for param in self.net.parameters():
            dist.broadcast(param.data, src=0)

def run_worker(rank, world_size, game, args):
    setup(rank, world_size)
    try:
        if rank == 0:
            parameter_server = ParameterServer(AlphaZeroNet(game, args))
            run_parameter_server(rank, world_size, parameter_server.model)
        else:
            alphazero = DistributedAlphaZero(game, args, rank, world_size)
            alphazero.distributed_learn()
    except Exception as e:
        print(f"Error in worker {rank}: {str(e)}")
    finally:
        if rank != 0:
            alphazero.logger.close()
        cleanup()

def distributed_main(game, args, num_workers):
    mp.spawn(run_worker, args=(num_workers + 1, game, args), nprocs=num_workers + 1, join=True)

if __name__ == "__main__":
    from .tictactoe import TicTacToe
    import argparse

    parser = argparse.ArgumentParser(description='Distributed AlphaZero')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for storing logs')
    args = parser.parse_args()

    game = TicTacToe()
    distributed_main(game, args, args.num_workers)
