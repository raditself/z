
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import chess
from src.alphazero.endgame_tablebase import EndgameTablebase

def test_endgame_tablebase():
    # Initialize the endgame tablebase
    tablebase = EndgameTablebase()

    # Test position: KPK endgame
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")

    # Test WDL probe
    wdl = tablebase.probe_wdl(board)
    print(f"WDL for KPK position: {wdl}")
    assert wdl in [-2, 0, 2], "WDL probe failed"

    # Test DTZ probe
    dtz = tablebase.probe_dtz(board)
    print(f"DTZ for KPK position: {dtz}")
    assert 1 <= dtz <= 50, "DTZ probe failed"

    # Test get_best_move
    best_move = tablebase.get_best_move(board)
    print(f"Best move for KPK position: {best_move}")
    assert best_move in board.legal_moves, "get_best_move failed"

    print("All tests passed!")

if __name__ == "__main__":
    test_endgame_tablebase()
