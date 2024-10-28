
import chess
import numpy as np
from src.alphazero.chess_ai import ChessAI
from src.alphazero.model import AlphaZeroModel
from src.alphazero.curriculum_learning import CurriculumLearning
from src.alphazero.game_phase_handler import GamePhaseHandler

class SelfPlay:
    def __init__(self, model_path, args):
        self.model = AlphaZeroModel(initial_learning_rate=args.get('initial_learning_rate', 0.001),
                                    decay_steps=args.get('decay_steps', 10000),
                                    decay_rate=args.get('decay_rate', 0.96))
        self.game_phase_handler = GamePhaseHandler()
        self.chess_ai = ChessAI(model_path, args, self.game_phase_handler)
        self.curriculum = CurriculumLearning()
        self.args = args

    def play_game(self):
        board = chess.Board()
        states, policies, values, game_phases = [], [], [], []

        while not board.is_game_over():
            # Apply curriculum learning
            board = self.curriculum.adjust_game_complexity(board)

            state = self.chess_ai.board_to_state(board)
            game_phase = self.game_phase_handler.calculate_game_phase(board)
            move, value = self.chess_ai.get_move(board.fen(), game_phase)

            # Store the state, policy, and game phase
            states.append(state)
            policy = np.zeros(4672)  # Total number of possible moves in chess
            policy[self.move_to_index(move)] = 1
            policies.append(policy)
            game_phases.append(game_phase)

            # Make the move
            board.push(move)

            # Store the value from the current player's perspective
            values.append(value if board.turn == chess.WHITE else -value)

        # Game result
        if board.result() == '1-0':
            game_result = 1
        elif board.result() == '0-1':
            game_result = -1
        else:
            game_result = 0

        # Update values based on the game result
        for i in range(len(values)):
            if i % 2 == 0:
                values[i] = game_result
            else:
                values[i] = -game_result

        return states, policies, values, game_phases

    def generate_training_data(self, num_games):
        all_states, all_policies, all_values, all_game_phases = [], [], [], []

        for _ in range(num_games):
            states, policies, values, game_phases = self.play_game()
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
            all_game_phases.extend(game_phases)
            # Step the curriculum after each game
            self.curriculum.step()

        return np.array(all_states), np.array(all_policies), np.array(all_values), np.array(all_game_phases)

    def train(self, num_games, batch_size=32, epochs=10):
        states, policies, values, game_phases = self.generate_training_data(num_games)
        self.model.train(states, policies, values, game_phases, batch_size=batch_size, epochs=epochs)
        # Reset the curriculum after training
        self.curriculum.reset()

    def move_to_index(self, move):
        # Convert a chess move to an index in the policy vector
        # This is a simplified version and might need to be expanded
        return move.from_square * 64 + move.to_square

# Example usage:
# self_play = SelfPlay('path_to_model_weights.h5', initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.96)
# self_play.train(num_games=100)
