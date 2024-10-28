
import numpy as np
from src.alphazero.game import Game

class ImperfectInfoGame(Game):
    def __init__(self):
        super().__init__()
        self.hidden_state = None

    def get_observation(self, player):
        # Return a partial observation of the game state
        raise NotImplementedError

    def get_legal_actions(self, player):
        # Return legal actions based on the partial observation
        raise NotImplementedError

    def step(self, action):
        # Update both the visible and hidden state
        raise NotImplementedError

    def is_terminal(self):
        # Check if the game has ended based on the full state
        raise NotImplementedError

    def get_reward(self, player):
        # Return the reward for the player based on the full state
        raise NotImplementedError

class PokerGame(ImperfectInfoGame):
    def __init__(self, num_players=2):
        super().__init__()
        self.num_players = num_players
        self.deck = list(range(52))
        self.player_hands = [[] for _ in range(num_players)]
        self.community_cards = []
        self.current_player = 0
        self.pot = 0
        self.player_bets = [0] * num_players
        self.hidden_state = {'deck': self.deck.copy()}

    def get_observation(self, player):
        return {
            'hand': self.player_hands[player],
            'community_cards': self.community_cards,
            'pot': self.pot,
            'player_bets': self.player_bets,
            'current_player': self.current_player
        }

    def get_legal_actions(self, player):
        if player != self.current_player:
            return []
        actions = ['fold', 'call', 'raise']
        return actions

    def step(self, action):
        if action == 'fold':
            self.current_player = (self.current_player + 1) % self.num_players
        elif action == 'call':
            max_bet = max(self.player_bets)
            self.pot += max_bet - self.player_bets[self.current_player]
            self.player_bets[self.current_player] = max_bet
            self.current_player = (self.current_player + 1) % self.num_players
        elif action == 'raise':
            raise_amount = 10  # Simplified raise amount
            self.pot += raise_amount
            self.player_bets[self.current_player] += raise_amount
            self.current_player = (self.current_player + 1) % self.num_players

        # Deal community cards if necessary
        if len(self.community_cards) < 5 and all(bet == self.player_bets[0] for bet in self.player_bets):
            self.deal_community_card()

    def deal_community_card(self):
        card = self.hidden_state['deck'].pop()
        self.community_cards.append(card)

    def is_terminal(self):
        return len(self.community_cards) == 5 and all(bet == self.player_bets[0] for bet in self.player_bets)

    def get_reward(self, player):
        if not self.is_terminal():
            return 0
        # Simplified poker hand evaluation
        hand_values = [self.evaluate_hand(player) for player in range(self.num_players)]
        winner = np.argmax(hand_values)
        return self.pot if player == winner else -self.player_bets[player]

    def evaluate_hand(self, player):
        # Simplified hand evaluation (just sum of card values)
        return sum(card % 13 for card in self.player_hands[player] + self.community_cards)
