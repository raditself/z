
import torch
import torch.distributed as dist
from .alphazero_net import AlphaZeroNet
from .game import Game
from argparse import Namespace

class ParameterServer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def update_model(self, gradients):
        for param, grad in zip(self.model.parameters(), gradients):
            if param.grad is None:
                param.grad = grad.clone().detach()
            else:
                param.grad += grad
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_model_params(self):
        return [param.data for param in self.model.parameters()]

def run_parameter_server(rank, world_size, model):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    server = ParameterServer(model)

    while True:
        # Wait for gradient updates from workers
        gradients = [torch.zeros_like(param) for param in model.parameters()]
        for i in range(len(gradients)):
            dist.recv(gradients[i], src=dist.get_world_size() - 1)
        
        # Update the model
        server.update_model(gradients)

        # Send updated model parameters to workers
        params = server.get_model_params()
        for i in range(len(params)):
            dist.broadcast(params[i], src=0)

if __name__ == "__main__":
    # These would typically be passed as arguments or loaded from a config file
    game = Game()  # You'll need to implement this or import from your game module
    args = Namespace(
        num_channels=256,
        num_residual_blocks=19,
        # Add other necessary arguments here
    )
    
    # Initialize the parameter server
    model = AlphaZeroNet(game, args)
    run_parameter_server(0, dist.get_world_size(), model)
