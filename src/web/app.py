
from flask import Flask, render_template, request, jsonify
from src.alphazero.game import ChessGame
from src.alphazero.model import create_model
from src.alphazero.mcts import MCTS
import torch
import numpy as np

app = Flask(__name__)

game = ChessGame()
model = create_model(board_size=8, action_size=game.action_size)
mcts = MCTS(game, model, num_processes=4)  # Use 4 processes for MCTS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_game_state')
def get_game_state():
    return jsonify({
        'board': game.board.tolist(),
        'current_player': game.current_player,
        'game_over': game.is_game_over(),
        'winner': game.get_winner() if game.is_game_over() else None
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    try:
        move = request.json['move']
        from_row, from_col, to_row, to_col = move
        game.make_move((from_row, from_col, to_row, to_col))
        
        if not game.is_game_over():
            # AI move
            state = torch.FloatTensor(game.get_state()[0]).unsqueeze(0)
            action_probs = mcts.get_action_prob(state, temp=1)
            action = np.random.choice(game.action_size, p=action_probs)
            from_row, from_col = action // 64 // 8, action // 64 % 8
            to_row, to_col = action % 64 // 8, action % 64 % 8
            game.make_move((from_row, from_col, to_row, to_col))
        
        return jsonify({
            'board': game.board.tolist(),
            'current_player': game.current_player,
            'game_over': game.is_game_over(),
            'winner': game.get_winner() if game.is_game_over() else None
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/reset_game', methods=['POST'])
def reset_game():
    global game
    game = ChessGame()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
