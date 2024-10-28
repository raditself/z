
from flask import Flask, render_template, request, jsonify
from src.games.chess import Chess
from src.alphazero.mcts import AdaptiveMCTS
from src.alphazero.neural_network import DynamicNeuralNetwork

app = Flask(__name__)

game = Chess()
network = DynamicNeuralNetwork(game)
mcts = AdaptiveMCTS(game, network)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/make_move', methods=['POST'])
def make_move():
    board_state = request.json['board_state']
    action = mcts.search(board_state)
    new_state = game.get_next_state(board_state, action)
    return jsonify({
        'action': action,
        'new_state': new_state.tolist()
    })

@app.route('/get_ai_explanation', methods=['POST'])
def get_ai_explanation():
    board_state = request.json['board_state']
    action = request.json['action']
    explanation = explainable_ai.explain_decision(board_state, action)
    return jsonify({'explanation': explanation})

if __name__ == '__main__':
    app.run(debug=True)
