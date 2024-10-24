import numpy as np
from .game import ChessGame

def analyze_game(game_history):
    analysis = []
    current_game = ChessGame()
    
    for move_number, move in enumerate(game_history, start=1):
        current_game.make_move(move)
        
        # Analyze every 5th move or the last move
        if move_number % 5 == 0 or move_number == len(game_history):
            evaluation = current_game.evaluate()
            key_position = is_key_position(current_game, evaluation)
            
            if key_position:
                analysis.append({
                    'move_number': move_number,
                    'position': current_game.board.copy(),
                    'evaluation': evaluation,
                    'suggested_move': get_best_move(current_game)
                })
    
    return analysis

def is_key_position(game, evaluation):
    # Define criteria for key positions
    # For example, significant change in evaluation, or critical moments in the game
    return abs(evaluation) > 300 or game.is_check()

def get_best_move(game):
    best_eval = float('-inf') if game.current_player == 1 else float('inf')
    best_move = None
    
    for move in game.get_legal_moves():
        new_game = game.clone()
        new_game.make_move(move)
        eval = new_game.evaluate()
        
        if game.current_player == 1:  # White's turn
            if eval > best_eval:
                best_eval = eval
                best_move = move
        else:  # Black's turn
            if eval < best_eval:
                best_eval = eval
                best_move = move
    
    return best_move

def suggest_improvements(analysis):
    suggestions = []
    for position in analysis:
        if position['evaluation'] < -100:  # Disadvantageous position for the current player
            suggestions.append({
                'move_number': position['move_number'],
                'suggestion': f"Consider {position['suggested_move']} instead of the played move."
            })
    return suggestions
