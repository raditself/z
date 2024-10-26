
import torch
import torch.nn.functional as F
import numpy as np
import random
from .alphazero_net import AlphaZeroNet
from .mcts import MCTS
from .parallel_self_play import ParallelSelfPlay

class AlphaZero:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AlphaZeroNet(game.board_size, game.action_size).to(self.device)
        self.mcts = MCTS(game, self.net, args)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.parallel_self_play = ParallelSelfPlay(game, args)

    def self_play(self, num_games):
        return self.parallel_self_play.parallel_self_play(num_games)

    def train(self, examples):
        self.net.train()
        for _ in range(self.args.epochs):
            batch = random.sample(examples, min(self.args.batch_size, len(examples)))
            state, policy_targets, value_targets = zip(*batch)

            state = torch.FloatTensor(np.array(state)).to(self.device)
            policy_targets = torch.FloatTensor(np.array(policy_targets)).to(self.device)
            value_targets = torch.FloatTensor(np.array(value_targets).astype(np.float64)).to(self.device)

            out_policy, out_value = self.net(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value.squeeze(-1), value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args.num_iterations):
            examples = self.self_play(self.args.num_episodes)

            self.train(examples)

            if iteration % self.args.checkpoint_interval == 0:
                torch.save(self.net.state_dict(), f'checkpoint_{iteration}.pth')

    def play(self, states):
        self.net.eval()
        with torch.no_grad():
            if not isinstance(states, list):
                states = [states]
            actions = self.mcts.search(states)
            return actions if len(actions) > 1 else actions[0]
