
import torch
import torch.nn.functional as F
import numpy as np
import random
from .alphazero_net import AlphaZeroNet
from .mcts import MCTS

class AlphaZero:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AlphaZeroNet(game.board_size, game.action_size).to(self.device)
        self.mcts = MCTS(game, self.net, args)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def self_play(self):
        memory = []
        state = self.game.get_initial_state()
        player = 1

        while True:
            canonical_state = self.game.get_canonical_form(state, player)
            action_probs = self.mcts.search(canonical_state)

            memory.append((canonical_state, action_probs, player))

            action = np.random.choice(self.game.action_size, p=action_probs)
            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                return_memory = []
                for hist_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value
                    return_memory.append((hist_state, hist_action_probs, hist_outcome))
                return return_memory

            player = self.game.get_opponent(player)

    def train(self, examples):
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
            examples = []
            for episode in range(self.args.num_episodes):
                examples += self.self_play()

            self.train(examples)

            if iteration % self.args.checkpoint_interval == 0:
                torch.save(self.net.state_dict(), f'checkpoint_{iteration}.pth')

    def play(self, state):
        with torch.no_grad():
            pi = self.mcts.search(state)
            return np.argmax(pi)
