
import sys
import os
import chess
import numpy as np

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.alphazero.mcts import MCTS
from src.alphazero.game import Game
from src.alphazero.neural_architecture_search import NeuralArchitectureSearch

class DummyArgs:
    def __init__(self):
        self.numMCTSSims = 100
        self.cpuct = 1.0
        self.EPS = 1e-8

def test_mcts_endgame_integration():
    game = Game()
    args = DummyArgs()
    nnet = NeuralArchitectureSearch(game)
    
    # Test endgame position (KPK)
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    canonical_board = game.getCanonicalForm(board, 1)
    
    mcts = MCTS(game, args, nnet)
    
    # Test if MCTS uses endgame tablebase for this position
    action_probs = mcts.getActionProb(canonical_board, temp=1)
    
    # The best move should be to push the pawn
    best_move = np.argmax(action_probs)
    assert game.actionToMove(board, best_move) == chess.Move.from_uci("e2e4"), "MCTS should choose to push the pawn in this KPK endgame"

    # Test a more complex endgame position (KBNK)
    board = chess.Board("4k3/8/8/8/8/8/4BN2/4K3 w - - 0 1")
    canonical_board = game.getCanonicalForm(board, 1)
    
    action_probs = mcts.getActionProb(canonical_board, temp=1)
    
    # The best moves should be to move the king towards the opponent's king
    best_moves = [chess.Move.from_uci(m) for m in ["e1d2", "e1f2", "e1d1", "e1f1"]]
    assert game.actionToMove(board, np.argmax(action_probs)) in best_moves, "MCTS should choose to move the king towards the opponent in this KBNK endgame"

    print("All endgame integration tests passed!")

if __name__ == "__main__":
    test_mcts_endgame_integration()
