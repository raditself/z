
import time

class GameTimer:
    def __init__(self, initial_time, increment=0):
        self.initial_time = initial_time
        self.increment = increment
        self.reset()

    def reset(self):
        self.time_left = {1: self.initial_time, -1: self.initial_time}
        self.last_move_time = {1: None, -1: None}

    def start_move(self, player):
        self.last_move_time[player] = time.time()

    def end_move(self, player):
        if self.last_move_time[player] is not None:
            elapsed_time = time.time() - self.last_move_time[player]
            self.time_left[player] -= elapsed_time
            self.time_left[player] += self.increment
            self.last_move_time[player] = None

    def get_time_left(self, player):
        if self.last_move_time[player] is not None:
            elapsed_time = time.time() - self.last_move_time[player]
            return max(0, self.time_left[player] - elapsed_time)
        return max(0, self.time_left[player])

    def is_flag_fallen(self, player):
        return self.get_time_left(player) <= 0

class GameWithTimer:
    def __init__(self, game, initial_time, increment=0):
        self.game = game
        self.timer = GameTimer(initial_time, increment)

    def make_move(self, board, move):
        current_player = self.game.get_current_player(board)
        self.timer.start_move(current_player)
        new_board = self.game.make_move(board, move)
        self.timer.end_move(current_player)
        return new_board

    def is_game_over(self, board):
        return self.game.is_game_over(board) or self.timer.is_flag_fallen(1) or self.timer.is_flag_fallen(-1)

    def get_winner(self, board):
        if self.timer.is_flag_fallen(1):
            return -1
        elif self.timer.is_flag_fallen(-1):
            return 1
        else:
            return self.game.get_winner(board)

    def get_time_left(self, player):
        return self.timer.get_time_left(player)

    # Delegate other methods to the original game object
    def __getattr__(self, name):
        return getattr(self.game, name)

# Usage example:
# from src.games.chess import ChessGame
# from src.games.checkers import CheckersGame

# chess_game_with_timer = GameWithTimer(ChessGame(), initial_time=600, increment=10)  # 10 minutes + 10 seconds increment
# checkers_game_with_timer = GameWithTimer(CheckersGame(), initial_time=300, increment=5)  # 5 minutes + 5 seconds increment

# To use in the main game loop:
# while not game_with_timer.is_game_over(board):
#     current_player = game_with_timer.get_current_player(board)
#     print(f"Time left for player {current_player}: {game_with_timer.get_time_left(current_player):.1f} seconds")
#     move = get_move_from_player_or_ai(board)
#     board = game_with_timer.make_move(board, move)

# winner = game_with_timer.get_winner(board)
# print(f"Game over. Winner: {winner}")
