
from flask import Flask, render_template, request, jsonify
from src.alphazero.game import ChessGame
from src.alphazero.model import create_model
from src.alphazero.mcts import MCTS
from src.alphazero.train import TrainAlphaZero
import torch
import numpy as np
import threading
import time

app = Flask(__name__)

game = ChessGame()
model = create_model(board_size=8, action_size=game.action_size)
mcts = MCTS(game, model, num_processes=4)  # Use 4 processes for MCTS

# Global variables for training
trainer = None
training_thread = None
is_training = False
training_progress = {'iteration': 0, 'win_rate': 0, 'loss_rate': 0, 'draw_rate': 0}

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

@app.route('/start_training', methods=['POST'])
def start_training():
    global trainer, training_thread, is_training
    if not is_training:
        trainer = TrainAlphaZero(game, model, num_iterations=1000, num_episodes=10)
        is_training = True
        training_thread = threading.Thread(target=train_model)
        training_thread.start()
        return jsonify({'success': True, 'message': 'Training started'})
    else:
        return jsonify({'success': False, 'message': 'Training is already in progress'})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global is_training
    if is_training:
        is_training = False
        return jsonify({'success': True, 'message': 'Training stopped'})
    else:
        return jsonify({'success': False, 'message': 'No training in progress'})

@app.route('/get_training_progress')
def get_training_progress():
    global training_progress
    return jsonify(training_progress)

def train_model():
    global training_progress, is_training
    while is_training:
        iteration, win_rate, loss_rate, draw_rate = trainer.train_iteration()
        training_progress = {
            'iteration': iteration,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'draw_rate': draw_rate
        }
        time.sleep(1)  # Add a small delay to prevent excessive CPU usage

@app.route('/play_against_ai', methods=['POST'])
def play_against_ai():
    try:
        state = torch.FloatTensor(game.get_state()[0]).unsqueeze(0)
        action_probs = mcts.get_action_prob(state, temp=0.1)  # Lower temperature for stronger play
        action = np.random.choice(game.action_size, p=action_probs)
        from_row, from_col = action // 64 // 8, action // 64 % 8
        to_row, to_col = action % 64 // 8, action % 64 % 8
        game.make_move((from_row, from_col, to_row, to_col))
        
        return jsonify({
            'board': game.board.tolist(),
            'current_player': game.current_player,
            'game_over': game.is_game_over(),
            'winner': game.get_winner() if game.is_game_over() else None,
            'ai_move': [from_row, from_col, to_row, to_col]
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
