
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from src.games.chess import Chess
from src.games.go import Go
from src.games.shogi import Shogi
from src.games.othello import Othello
from src.games.connect_four import ConnectFour
from src.alphazero.mcts import AdaptiveMCTS
from src.alphazero.neural_network import DynamicNeuralNetwork
from src.advanced_heuristics import AdvancedHeuristics
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app)

games = {
    'chess': Chess(),
    'go': Go(),
    'shogi': Shogi(),
    'othello': Othello(),
    'connect_four': ConnectFour()
}

networks = {game_name: DynamicNeuralNetwork(game) for game_name, game in games.items()}
mcts_agents = {game_name: AdaptiveMCTS(game, network) for game_name, (game, network) in zip(games.items(), networks.items())}

@app.route('/')
def index():
    return render_template('index.html', games=list(games.keys()))

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('select_game')
def handle_select_game(game_name):
    if game_name in games:
        emit('game_selected', {'game': game_name, 'initial_state': games[game_name].get_initial_state().tolist()})
    else:
        emit('error', {'message': 'Invalid game selection'})

@socketio.on('make_move')
def handle_make_move(data):
    game_name = data['game']
    move = data['move']
    game = games[game_name]
    mcts = mcts_agents[game_name]
    
    # Player's move
    state = game.get_next_state(game.board, move)
    game.board = state
    
    if not game.is_terminal(state):
        # AI's move
        ai_move = mcts.search(state)
        state = game.get_next_state(state, ai_move)
        game.board = state
        
        # Get heuristic evaluation
        heuristic_value = getattr(AdvancedHeuristics, f"{game_name}_heuristics")(state)
        
        emit('move_made', {
            'player_move': move,
            'ai_move': ai_move,
            'new_state': state.tolist(),
            'heuristic_value': heuristic_value,
            'is_terminal': game.is_terminal(state),
            'reward': game.get_reward(state) if game.is_terminal(state) else None
        })
    else:
        emit('game_over', {
            'final_state': state.tolist(),
            'reward': game.get_reward(state)
        })

def mcts_background_training():
    while True:
        for game_name, mcts in mcts_agents.items():
            mcts.search(games[game_name].get_initial_state())
        time.sleep(1)  # Adjust sleep time as needed

if __name__ == '__main__':
    training_thread = threading.Thread(target=mcts_background_training)
    training_thread.start()
    socketio.run(app, debug=True)
