
import { Chess } from 'chess.js';

class ChessAI {
  constructor(depth = 3) {
    this.depth = depth;
    this.game = new Chess();
  }

  makeMove() {
    const possibleMoves = this.game.moves();
    
    if (this.game.game_over() || possibleMoves.length === 0) {
      return null;
    }

    let bestMove = null;
    let bestValue = -Infinity;

    for (let move of possibleMoves) {
      this.game.move(move);
      const value = this.minimax(this.depth - 1, false, -Infinity, Infinity);
      this.game.undo();

      if (value > bestValue) {
        bestValue = value;
        bestMove = move;
      }
    }

    return bestMove;
  }

  minimax(depth, isMaximizingPlayer, alpha, beta) {
    if (depth === 0) {
      return this.evaluateBoard();
    }

    const possibleMoves = this.game.moves();

    if (isMaximizingPlayer) {
      let bestValue = -Infinity;
      for (let move of possibleMoves) {
        this.game.move(move);
        bestValue = Math.max(bestValue, this.minimax(depth - 1, false, alpha, beta));
        this.game.undo();
        alpha = Math.max(alpha, bestValue);
        if (beta <= alpha) {
          break;
        }
      }
      return bestValue;
    } else {
      let bestValue = Infinity;
      for (let move of possibleMoves) {
        this.game.move(move);
        bestValue = Math.min(bestValue, this.minimax(depth - 1, true, alpha, beta));
        this.game.undo();
        beta = Math.min(beta, bestValue);
        if (beta <= alpha) {
          break;
        }
      }
      return bestValue;
    }
  }

  evaluateBoard() {
    let totalEvaluation = 0;
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        totalEvaluation += this.getPieceValue(this.game.get(String.fromCharCode(97 + i) + (j + 1)));
      }
    }
    return totalEvaluation;
  }

  getPieceValue(piece) {
    if (piece === null) {
      return 0;
    }
    const pieceValue = {
      'p': 10,
      'n': 30,
      'b': 30,
      'r': 50,
      'q': 90,
      'k': 900
    };
    return pieceValue[piece.type] * (piece.color === 'w' ? 1 : -1);
  }

  setPosition(fen) {
    this.game.load(fen);
  }

  getGame() {
    return this.game;
  }
}

export default ChessAI;
