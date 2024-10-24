import json
import numpy as np
from .game import ChessGame

def save_game_state(game, filename):
    state = {
        'board': game.board.tolist(),
        'current_player': game.current_player,
        'castling_rights': game.castling_rights,
        'en_passant_target': game.en_passant_target,
        'halfmove_clock': game.halfmove_clock,
        'fullmove_number': game.fullmove_number
    }
    with open(filename, 'w') as f:
        json.dump(state, f)

def load_game_state(filename):
    with open(filename, 'r') as f:
        state = json.load(f)
    
    game = ChessGame()
    game.board = np.array(state['board'])
    game.current_player = state['current_player']
    game.castling_rights = state['castling_rights']
    game.en_passant_target = state['en_passant_target']
    game.halfmove_clock = state['halfmove_clock']
    game.fullmove_number = state['fullmove_number']
    
    return game
