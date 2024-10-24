
from flask import Flask, render_template, request, jsonify
from src.alphazero.game import ChessGame
from src.alphazero.model import create_model
from src.alphazero.mcts import MCTS
import torch

app = Flask(__name__)

game = ChessGame()
model = create_model()
mcts = MCTS(game, model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/make_move', methods=['POST'])
def make_move():
    move = request.json['move']
    game.make_move(move)
    
    if game.is_game_over():
        return jsonify({'game_over': True, 'winner': game.get_winner()})
    
    # AI move
    state = torch.FloatTensor(game.get_state()[0]).unsqueeze(0)
    mcts_probs = mcts.search(state)
    action = max(range(len(mcts_probs)), key=mcts_probs.__getitem__)
    game.make_move(action)
    
    return jsonify({
        'board': game.board.tolist(),
        'current_player': game.current_player,
        'game_over': game.is_game_over(),
        'winner': game.get_winner() if game.is_game_over() else None
    })

@app.route('/reset_game', methods=['POST'])
def reset_game():
    global game
    game = ChessGame()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
