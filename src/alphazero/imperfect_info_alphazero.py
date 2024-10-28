
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import multiprocessing as mp
import math

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

class ImperfectInfoGame:
    def __init__(self):
        self.action_size = None
        self.num_players = None
        self.current_player = 0
        self.hidden_state = None

    def get_initial_state(self):
        raise NotImplementedError

    def get_next_state(self, state, action):
        raise NotImplementedError

    def get_valid_moves(self, state):
        raise NotImplementedError

    def get_game_ended(self, state, player):
        raise NotImplementedError

    def get_observation(self, state, player):
        raise NotImplementedError

    def get_current_player(self, state):
        return state['current_player']

class PokerGame(ImperfectInfoGame):
    def __init__(self, num_players=2):
        super().__init__()
        self.num_players = num_players
        self.action_size = 3  # fold, call, raise
        self.deck = list(range(52))
        self.small_blind = 1
        self.big_blind = 2

    def get_initial_state(self):
        random.shuffle(self.deck)
        hands = [self.deck[i:i+2] for i in range(0, 2*self.num_players, 2)]
        community_cards = []
        pot = self.small_blind + self.big_blind
        current_player = 2 % self.num_players
        bets = [0] * self.num_players
        bets[0] = self.small_blind
        bets[1] = self.big_blind
        return {
            'hands': hands,
            'community_cards': community_cards,
            'pot': pot,
            'current_player': current_player,
            'bets': bets,
            'deck': self.deck[2*self.num_players:],
            'folded': [False] * self.num_players
        }

    def get_next_state(self, state, action):
        new_state = state.copy()
        player = new_state['current_player']
        if action == 0:  # fold
            new_state['folded'][player] = True
        elif action == 1:  # call
            max_bet = max(new_state['bets'])
            new_state['pot'] += max_bet - new_state['bets'][player]
            new_state['bets'][player] = max_bet
        elif action == 2:  # raise
            raise_amount = 2 * max(new_state['bets'])
            new_state['pot'] += raise_amount - new_state['bets'][player]
            new_state['bets'][player] = raise_amount

        # Move to next player
        new_state['current_player'] = (player + 1) % self.num_players
        while new_state['folded'][new_state['current_player']]:
            new_state['current_player'] = (new_state['current_player'] + 1) % self.num_players

        # Check if round is over and deal community cards if necessary
        if all(bet == max(new_state['bets']) for bet in new_state['bets']) and not all(new_state['folded']):
            if len(new_state['community_cards']) == 0:
                new_state['community_cards'] = new_state['deck'][:3]
                new_state['deck'] = new_state['deck'][3:]
            elif len(new_state['community_cards']) == 3:
                new_state['community_cards'].append(new_state['deck'].pop(0))
            elif len(new_state['community_cards']) == 4:
                new_state['community_cards'].append(new_state['deck'].pop(0))

        return new_state

    def get_valid_moves(self, state):
        return [1, 1, 1]  # Always allow fold, call, raise for simplicity

    def get_game_ended(self, state, player):
        if sum(state['folded']) == self.num_players - 1:
            return 1 if not state['folded'][player] else -1
        if len(state['community_cards']) == 5 and all(bet == max(state['bets']) for bet in state['bets']):
            # Simplified scoring, just count the highest card
            scores = [max(hand + state['community_cards']) for hand in state['hands']]
            winner = scores.index(max(scores))
            return 1 if winner == player else -1
        return 0

    def get_observation(self, state, player):
        return {
            'hand': state['hands'][player],
            'community_cards': state['community_cards'],
            'pot': state['pot'],
            'bets': state['bets'],
            'current_player': state['current_player'],
            'folded': state['folded']
        }

class ImperfectInfoAlphaZeroNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        self.args = args
        self.action_size = game.action_size
        self.num_players = game.num_players
        
        self.conv1 = nn.Conv2d(3, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        
        self.fc1 = nn.Linear(args.num_channels * (args.board_x - 2) * (args.board_y - 2), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        
        self.pi_head = nn.Linear(512, game.action_size)
        self.v_head = nn.Linear(512, 1)
        
    def forward(self, s):
        s = s.view(-1, 3, self.args.board_x, self.args.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = s.view(-1, self.args.num_channels * (self.args.board_x - 2) * (self.args.board_y - 2))
        
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)
        
        pi = self.pi_head(s)
        v = self.v_head(s)
        
        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, state):
        self.eval()
        with torch.no_grad():
            pi, v = self(self.state_to_tensor(state))
        return pi.exp().cpu().numpy()[0], v.cpu().numpy()[0]

    def state_to_tensor(self, state):
        # Convert state to tensor representation
        # This is a placeholder implementation and should be adapted to your specific game representation
        return torch.FloatTensor(state)

class ImperfectInfoMCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}

    def search(self, state):
        s = self.game.get_observation(state, state['current_player'])
        if self.game.get_game_ended(state, state['current_player']):
            return -self.game.get_game_ended(state, state['current_player'])

        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(s)
            valids = self.game.get_valid_moves(state)
            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.action_size):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.game.get_next_state(state, a)
        next_player = self.game.get_current_player(next_s)
        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def get_action_prob(self, state, temp=1):
        for _ in range(self.args.num_mcts_sims):
            self.search(state)

        s = self.game.get_observation(state, state['current_player'])
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.action_size)]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        return probs

class ImperfectInfoAlphaZero:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = ImperfectInfoMCTS(game, nnet, args)

    def learn(self):
        for i in range(self.args.num_iters):
            print(f'Starting Iter #{i} ...')
            train_examples = []
            for _ in range(self.args.num_eps):
                train_examples.extend(self.execute_episode())
            
            self.nnet.train(train_examples)
            self.mcts = ImperfectInfoMCTS(self.game, self.nnet, self.args)

    def execute_episode(self):
        train_examples = []
        state = self.game.get_initial_state()
        current_player = self.game.get_current_player(state)

        while True:
            temp = int(self.args.num_moves_for_tau0 > len(train_examples))
            pi = self.mcts.get_action_prob(state, temp)
            sym = self.game.get_observation(state, current_player)
            train_examples.append([sym, current_player, pi, None])
            
            action = np.random.choice(len(pi), p=pi)
            state = self.game.get_next_state(state, action)
            current_player = self.game.get_current_player(state)
            
            r = self.game.get_game_ended(state, current_player)
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != current_player))) for x in train_examples]

# Usage
args = dotdict({
    'num_iters': 1000,
    'num_eps': 100,
    'num_mcts_sims': 25,
    'cpuct': 1.0,
    'num_moves_for_tau0': 10,
    'arena_compare': 40,
    'lr': 0.001,
    'dropout': 0.3,
    'num_channels': 512,
    'board_x': 10,
    'board_y': 10,
})

game = PokerGame()
nnet = ImperfectInfoAlphaZeroNet(game, args)
az = ImperfectInfoAlphaZero(game, nnet, args)
az.learn()
