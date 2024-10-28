
import chess
import numpy as np
from src.alphazero.model import AlphaZeroModel
from src.alphazero.adaptive_mcts import AdaptiveMCTS

class ChessAI:
    def __init__(self, model_path, args):
        self.model = AlphaZeroModel()
        self.model.load_weights(model_path)
        self.args = args
        self.mcts = AdaptiveMCTS(self.model, args)

    def get_move(self, fen):
        board = chess.Board(fen)
        game_phase = self.calculate_game_phase(board)
        
        # Perform AdaptiveMCTS
        action_probs, value = self.mcts.search(self.board_to_state(board), game_phase)
        
        # Choose move based on probabilities
        moves = list(board.legal_moves)
        probs = action_probs[list(map(self.move_to_index, moves))]
        move_idx = np.random.choice(len(moves), p=probs / np.sum(probs))
        chosen_move = moves[move_idx]
        
        # Visualize adaptive parameters after the move
        self.mcts.visualize_adaptive_parameters()
        
        return chosen_move, value

    def board_to_state(self, board):
        # Improved board representation
        state = np.zeros((8, 8, 14), dtype=np.float32)
        
        # Piece placement
        piece_channels = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                state[rank, file, piece_channels[piece.symbol()]] = 1
        
        # Side to move
        state[:, :, 12] = float(board.turn)
        
        # Move count
        state[:, :, 13] = board.fullmove_number / 100.0  # Normalize
        
        return state

    def move_to_index(self, move):
        # Convert a chess move to an index in the policy vector
        return move.from_square * 64 + move.to_square

    def calculate_game_phase(self, board):
        # Calculate the game phase based on the number of pieces on the board
        total_pieces = sum(len(board.pieces(piece_type, color)) 
                           for color in [chess.WHITE, chess.BLACK] 
                           for piece_type in range(1, 7))
        
        # Assuming 32 pieces at the start, and considering the game in 3 phases
        if total_pieces >= 28:  # More than 87.5% of pieces remaining
            return 0.0  # Opening
        elif total_pieces >= 10:  # More than 31.25% of pieces remaining
            return 0.5  # Middlegame
        else:
            return 1.0  # Endgame

    def play_game(self):
        board = chess.Board()
        while not board.is_game_over():
            move, _ = self.get_move(board.fen())
            board.push(move)
            print(f"Move: {move}")
        
        print(f"Game over. Result: {board.result()}")
        return board.result()

# Example usage:
# args = {'num_mcts_sims': 800, 'c_puct': 1.0, 'dirichlet_alpha': 0.3, 'dirichlet_epsilon': 0.25, 'base_c_puct': 1.0}
# ai = ChessAI('path_to_model_weights.h5', args)
# result = ai.play_game()
# print(f"Game result: {result}")
