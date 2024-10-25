
import numpy as np
from checkers import Checkers, CheckersAI

class CheckersSelfPlay:
    def __init__(self, num_games=1000, max_moves=200):
        self.num_games = num_games
        self.max_moves = max_moves
        self.game = Checkers()
        self.ai = CheckersAI(self.game)
        self.position_history = {}

    def run_self_play(self):
        for game in range(self.num_games):
            self.game = Checkers()  # Reset the game
            move_count = 0
            while not self.game.is_game_over() and move_count < self.max_moves:
                best_move = self.ai.get_best_move()
                self.update_position_history(self.game.get_state(), best_move)
                self.game.make_move(best_move)
                move_count += 1
            
            self.update_final_positions(self.game.get_winner())
            
            if game % 100 == 0:
                print(f"Completed {game} games")

        self.update_ai_from_history()

    def update_position_history(self, state, move):
        state_key = self.state_to_key(state)
        if state_key not in self.position_history:
            self.position_history[state_key] = {'count': 0, 'moves': {}}
        
        self.position_history[state_key]['count'] += 1
        move_key = self.move_to_key(move)
        if move_key not in self.position_history[state_key]['moves']:
            self.position_history[state_key]['moves'][move_key] = 0
        self.position_history[state_key]['moves'][move_key] += 1

    def update_final_positions(self, winner):
        for state_key in self.position_history:
            if winner == 1:
                self.position_history[state_key]['white_win'] = self.position_history[state_key].get('white_win', 0) + 1
            elif winner == 2:
                self.position_history[state_key]['black_win'] = self.position_history[state_key].get('black_win', 0) + 1
            else:
                self.position_history[state_key]['draw'] = self.position_history[state_key].get('draw', 0) + 1

    def update_ai_from_history(self):
        for state_key, data in self.position_history.items():
            total_games = data['white_win'] + data['black_win'] + data['draw']
            win_rate = (data['white_win'] + 0.5 * data['draw']) / total_games
            
            best_move = max(data['moves'], key=data['moves'].get)
            best_move_frequency = data['moves'][best_move] / data['count']
            
            # Update AI's evaluation function or move selection based on win_rate and best_move_frequency
            # This is a placeholder and should be implemented based on your specific AI architecture
            print(f"State: {state_key}, Win Rate: {win_rate}, Best Move: {best_move}, Frequency: {best_move_frequency}")

    @staticmethod
    def state_to_key(state):
        return ''.join(map(str, state.flatten()))

    @staticmethod
    def move_to_key(move):
        return '_'.join(map(str, move))

if __name__ == "__main__":
    self_play = CheckersSelfPlay(num_games=1000)
    self_play.run_self_play()
