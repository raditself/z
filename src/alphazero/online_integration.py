
import requests
import chess
import chess.pgn
import io
import json
import logging
import time
from .mcts import MCTS
from .game import Game
from .model import get_model
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LichessIntegration:
    def __init__(self, api_token):
        self.api_token = api_token
        self.base_url = "https://lichess.org/api/"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    # ... (keep existing methods)

    def check_for_challenges(self):
        url = f"{self.base_url}challenge"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            challenges = response.json()['in']
            if challenges:
                return challenges[0]['id']  # Return the ID of the first challenge
            return None
        except requests.RequestException as e:
            logger.error(f"Error checking for challenges: {e}")
            return None

    def get_ongoing_game(self):
        url = f"{self.base_url}account/playing"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            games = response.json()['nowPlaying']
            if games:
                return games[0]['gameId']  # Return the ID of the first ongoing game
            return None
        except requests.RequestException as e:
            logger.error(f"Error getting ongoing games: {e}")
            return None

class ChessComIntegration:
    # ... (keep existing code)

class OnlineIntegration:
    def __init__(self, game: Game, args, model):
        self.game = game
        self.args = args
        self.model = model
        self.mcts = MCTS(self.game, self.args, self.model)

        self.lichess = LichessIntegration(args.lichess_token)
        self.chess_com = ChessComIntegration(args.chess_com_username, args.chess_com_password)

    # ... (keep existing methods)

    def wait_for_challenges(self):
        while True:
            game_id = self.lichess.check_for_challenges()
            if game_id:
                logger.info(f"Accepting challenge: {game_id}")
                self.play_lichess_game(game_id)
            else:
                time.sleep(10)  # Wait for 10 seconds before checking again

    def continuous_analysis(self):
        while True:
            game_id = self.lichess.get_ongoing_game()
            if game_id:
                logger.info(f"Analyzing game: {game_id}")
                self.real_time_analysis(game_id)
            else:
                time.sleep(10)  # Wait for 10 seconds before checking again

# ... (keep existing code)

if __name__ == "__main__":
    from .utils import dotdict
    
    args = dotdict({
        'lichess_token': 'your_lichess_token_here',
        'chess_com_username': 'your_chess_com_username',
        'chess_com_password': 'your_chess_com_password',
        'numMCTSSims': 100,
        'cpuct': 1.0,
        'moveTime': 5,
    })

    game = Game()
    model = get_model()
    online_integration = OnlineIntegration(game, args, model)

    # Wait for and play Lichess games
    online_integration.wait_for_challenges()

    # Continuous real-time analysis of Lichess games
    online_integration.continuous_analysis()
