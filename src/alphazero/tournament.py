
import numpy as np
from src.alphazero.chess_ai import ChessAI
from src.alphazero.checkers_ai import CheckersAI
from src.games.chess import ChessGame
from src.games.checkers import CheckersGame

class Tournament:
    def __init__(self, game_type='chess'):
        self.game_type = game_type
        if game_type == 'chess':
            self.game = ChessGame()
            self.AI_class = ChessAI
        elif game_type == 'checkers':
            self.game = CheckersGame()
            self.AI_class = CheckersAI
        else:
            raise ValueError("Invalid game type. Choose 'chess' or 'checkers'.")

    def play_match(self, ai1, ai2, num_games=10):
        scores = [0, 0, 0]  # [ai1 wins, ai2 wins, draws]

        for i in range(num_games):
            board = self.game.get_initial_board()
            players = [ai1, ai2] if i % 2 == 0 else [ai2, ai1]
            
            while not self.game.is_game_over(board):
                current_player = players[self.game.get_current_player(board)]
                move = current_player.get_move(board, temperature=0.1)
                board = self.game.make_move(board, move)

            winner = self.game.get_winner(board)
            if winner == 1:
                scores[0 if i % 2 == 0 else 1] += 1
            elif winner == -1:
                scores[1 if i % 2 == 0 else 0] += 1
            else:
                scores[2] += 1

        return scores

    def run_tournament(self, model_paths, num_games_per_match=10):
        num_players = len(model_paths)
        results = np.zeros((num_players, num_players))

        for i in range(num_players):
            for j in range(i+1, num_players):
                ai1 = self.AI_class(model_paths[i])
                ai2 = self.AI_class(model_paths[j])
                
                scores = self.play_match(ai1, ai2, num_games_per_match)
                results[i, j] = scores[0]
                results[j, i] = scores[1]

        return results

    def print_results(self, results, model_names):
        print(f"Tournament Results ({self.game_type}):")
        print("+" + "-" * (9 * (len(model_names) + 1)) + "+")
        print("|{:8s}|".format(""), end="")
        for name in model_names:
            print("{:8s}|".format(name[:8]), end="")
        print()
        print("+" + "-" * (9 * (len(model_names) + 1)) + "+")

        for i, name in enumerate(model_names):
            print("|{:8s}|".format(name[:8]), end="")
            for j in range(len(model_names)):
                if i == j:
                    print("{:8s}|".format("-"), end="")
                else:
                    print("{:8.1f}|".format(results[i, j]), end="")
            print()

        print("+" + "-" * (9 * (len(model_names) + 1)) + "+")

        total_wins = np.sum(results, axis=1)
        print("Total Wins:")
        for name, wins in zip(model_names, total_wins):
            print(f"{name}: {wins}")

# Usage example:
# chess_tournament = Tournament('chess')
# chess_model_paths = ['chess_model_v1.h5', 'chess_model_v2.h5', 'chess_model_v3.h5']
# chess_results = chess_tournament.run_tournament(chess_model_paths)
# chess_tournament.print_results(chess_results, ['v1', 'v2', 'v3'])

# checkers_tournament = Tournament('checkers')
# checkers_model_paths = ['checkers_model_v1.h5', 'checkers_model_v2.h5', 'checkers_model_v3.h5']
# checkers_results = checkers_tournament.run_tournament(checkers_model_paths)
# checkers_tournament.print_results(checkers_results, ['v1', 'v2', 'v3'])
