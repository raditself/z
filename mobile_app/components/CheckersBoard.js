
import React, { useState, useEffect } from 'react';
import { View, StyleSheet, TouchableOpacity, Text } from 'react-native';

const BOARD_SIZE = 8;
const CELL_SIZE = 40;

const CheckersBoard = ({ difficulty }) => {
  const [board, setBoard] = useState(initializeBoard());
  const [selectedPiece, setSelectedPiece] = useState(null);
  const [currentPlayer, setCurrentPlayer] = useState('red');
  const [isThinking, setIsThinking] = useState(false);

  useEffect(() => {
    console.log(`Difficulty set to: ${difficulty}`);
  }, [difficulty]);

  function initializeBoard() {
    const board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(null));
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < BOARD_SIZE; j++) {
        if ((i + j) % 2 === 1) {
          board[i][j] = 'black';
        }
      }
    }
    for (let i = 5; i < BOARD_SIZE; i++) {
      for (let j = 0; j < BOARD_SIZE; j++) {
        if ((i + j) % 2 === 1) {
          board[i][j] = 'red';
        }
      }
    }
    return board;
  }

  function handleCellPress(row, col) {
    if (selectedPiece) {
      movePiece(row, col);
    } else if (board[row][col] === currentPlayer) {
      setSelectedPiece({ row, col });
    }
  }

  function movePiece(toRow, toCol) {
    const { row: fromRow, col: fromCol } = selectedPiece;
    if (isValidMove(fromRow, fromCol, toRow, toCol)) {
      const newBoard = [...board];
      newBoard[toRow][toCol] = currentPlayer;
      newBoard[fromRow][fromCol] = null;
      
      if (Math.abs(fromRow - toRow) === 2) {
        const capturedRow = (fromRow + toRow) / 2;
        const capturedCol = (fromCol + toCol) / 2;
        newBoard[capturedRow][capturedCol] = null;
      }

      setBoard(newBoard);
      setCurrentPlayer(currentPlayer === 'red' ? 'black' : 'red');
      setSelectedPiece(null);

      // Trigger AI move after player's move
      setTimeout(() => makeAIMove(newBoard), 500);
    } else {
      setSelectedPiece(null);
    }
  }

  function makeAIMove(currentBoard) {
    setIsThinking(true);
    const aiColor = currentPlayer;
    const possibleMoves = getAllPossibleMoves(currentBoard, aiColor);
    
    let bestMove;
    if (difficulty === 'easy') {
      bestMove = possibleMoves[Math.floor(Math.random() * possibleMoves.length)];
    } else {
      bestMove = getBestMove(currentBoard, aiColor, difficulty === 'hard' ? 3 : 1);
    }

    if (bestMove) {
      const newBoard = [...currentBoard];
      newBoard[bestMove.toRow][bestMove.toCol] = aiColor;
      newBoard[bestMove.fromRow][bestMove.fromCol] = null;
      
      if (Math.abs(bestMove.fromRow - bestMove.toRow) === 2) {
        const capturedRow = (bestMove.fromRow + bestMove.toRow) / 2;
        const capturedCol = (bestMove.fromCol + bestMove.toCol) / 2;
        newBoard[capturedRow][capturedCol] = null;
      }

      setBoard(newBoard);
      setCurrentPlayer(aiColor === 'red' ? 'black' : 'red');
    }
    setIsThinking(false);
  }

  function getAllPossibleMoves(board, color) {
    const moves = [];
    for (let row = 0; row < BOARD_SIZE; row++) {
      for (let col = 0; col < BOARD_SIZE; col++) {
        if (board[row][col] === color) {
          const possibleMoves = getPossibleMovesForPiece(board, row, col);
          moves.push(...possibleMoves);
        }
      }
    }
    return moves;
  }

  function getPossibleMovesForPiece(board, row, col) {
    const moves = [];
    const directions = board[row][col] === 'red' ? [-1, 1] : [1, -1];
    
    for (const colDir of directions) {
      if (isValidMove(row, col, row + 1, col + colDir)) {
        moves.push({ fromRow: row, fromCol: col, toRow: row + 1, toCol: col + colDir });
      }
      if (isValidMove(row, col, row + 2, col + 2 * colDir)) {
        moves.push({ fromRow: row, fromCol: col, toRow: row + 2, toCol: col + 2 * colDir });
      }
    }
    
    return moves;
  }

  function getBestMove(board, color, depth) {
    const possibleMoves = getAllPossibleMoves(board, color);
    let bestMove = null;
    let bestScore = color === 'red' ? -Infinity : Infinity;

    for (const move of possibleMoves) {
      const newBoard = [...board];
      newBoard[move.toRow][move.toCol] = color;
      newBoard[move.fromRow][move.fromCol] = null;
      
      if (Math.abs(move.fromRow - move.toRow) === 2) {
        const capturedRow = (move.fromRow + move.toRow) / 2;
        const capturedCol = (move.fromCol + move.toCol) / 2;
        newBoard[capturedRow][capturedCol] = null;
      }

      const score = minimax(newBoard, depth - 1, color === 'red' ? 'black' : 'red', -Infinity, Infinity);

      if (color === 'red' && score > bestScore) {
        bestScore = score;
        bestMove = move;
      } else if (color === 'black' && score < bestScore) {
        bestScore = score;
        bestMove = move;
      }
    }

    return bestMove;
  }

  function minimax(board, depth, color, alpha, beta) {
    if (depth === 0) {
      return evaluateBoard(board);
    }

    const possibleMoves = getAllPossibleMoves(board, color);

    if (color === 'red') {
      let maxScore = -Infinity;
      for (const move of possibleMoves) {
        const newBoard = [...board];
        newBoard[move.toRow][move.toCol] = color;
        newBoard[move.fromRow][move.fromCol] = null;
        
        if (Math.abs(move.fromRow - move.toRow) === 2) {
          const capturedRow = (move.fromRow + move.toRow) / 2;
          const capturedCol = (move.fromCol + move.toCol) / 2;
          newBoard[capturedRow][capturedCol] = null;
        }

        const score = minimax(newBoard, depth - 1, 'black', alpha, beta);
        maxScore = Math.max(maxScore, score);
        alpha = Math.max(alpha, score);
        if (beta <= alpha) {
          break;
        }
      }
      return maxScore;
    } else {
      let minScore = Infinity;
      for (const move of possibleMoves) {
        const newBoard = [...board];
        newBoard[move.toRow][move.toCol] = color;
        newBoard[move.fromRow][move.fromCol] = null;
        
        if (Math.abs(move.fromRow - move.toRow) === 2) {
          const capturedRow = (move.fromRow + move.toRow) / 2;
          const capturedCol = (move.fromCol + move.toCol) / 2;
          newBoard[capturedRow][capturedCol] = null;
        }

        const score = minimax(newBoard, depth - 1, 'red', alpha, beta);
        minScore = Math.min(minScore, score);
        beta = Math.min(beta, score);
        if (beta <= alpha) {
          break;
        }
      }
      return minScore;
    }
  }

  function evaluateBoard(board) {
    let score = 0;
    for (let row = 0; row < BOARD_SIZE; row++) {
      for (let col = 0; col < BOARD_SIZE; col++) {
        if (board[row][col] === 'red') {
          score += 1 + (row === 0 ? 0.5 : 0);
        } else if (board[row][col] === 'black') {
          score -= 1 + (row === BOARD_SIZE - 1 ? 0.5 : 0);
        }
      }
    }
    return score;
  }

  function isValidMove(fromRow, fromCol, toRow, toCol) {
    if (board[toRow][toCol] !== null) return false;
    const rowDiff = toRow - fromRow;
    const colDiff = toCol - fromCol;
    if (currentPlayer === 'red' && rowDiff > 0) return false;
    if (currentPlayer === 'black' && rowDiff < 0) return false;
    if (Math.abs(rowDiff) === 1 && Math.abs(colDiff) === 1) return true;
    if (Math.abs(rowDiff) === 2 && Math.abs(colDiff) === 2) {
      const capturedRow = (fromRow + toRow) / 2;
      const capturedCol = (fromCol + toCol) / 2;
      return board[capturedRow][capturedCol] === (currentPlayer === 'red' ? 'black' : 'red');
    }
    return false;
  }

  function resetGame() {
    setBoard(initializeBoard());
    setSelectedPiece(null);
    setCurrentPlayer('red');
    setIsThinking(false);
  }

  return (
    <View style={styles.container}>
      <Text style={styles.currentPlayer}>Current Player: {currentPlayer}</Text>
      <Text style={styles.difficulty}>Difficulty: {difficulty}</Text>
      {isThinking && <Text style={styles.thinking}>AI is thinking...</Text>}
      <View style={styles.board}>
        {board.map((row, rowIndex) => (
          <View key={rowIndex} style={styles.row}>
            {row.map((cell, colIndex) => (
              <TouchableOpacity
                key={colIndex}
                style={[
                  styles.cell,
                  { backgroundColor: (rowIndex + colIndex) % 2 === 0 ? '#fff' : '#000' },
                  selectedPiece?.row === rowIndex && selectedPiece?.col === colIndex && styles.selectedCell,
                ]}
                onPress={() => handleCellPress(rowIndex, colIndex)}
                disabled={isThinking || currentPlayer !== 'red'}
              >
                {cell && (
                  <View style={[styles.piece, { backgroundColor: cell }]} />
                )}
              </TouchableOpacity>
            ))}
          </View>
        ))}
      </View>
      <TouchableOpacity style={styles.resetButton} onPress={resetGame}>
        <Text style={styles.resetButtonText}>Reset Game</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  currentPlayer: {
    fontSize: 18,
    marginBottom: 10,
    fontWeight: 'bold',
  },
  difficulty: {
    fontSize: 16,
    marginBottom: 10,
  },
  thinking: {
    fontSize: 16,
    marginBottom: 10,
    color: 'blue',
    fontStyle: 'italic',
  },
  board: {
    borderWidth: 1,
    borderColor: '#000',
    marginBottom: 20,
  },
  row: {
    flexDirection: 'row',
  },
  cell: {
    width: CELL_SIZE,
    height: CELL_SIZE,
    justifyContent: 'center',
    alignItems: 'center',
  },
  piece: {
    width: CELL_SIZE * 0.8,
    height: CELL_SIZE * 0.8,
    borderRadius: CELL_SIZE * 0.4,
  },
  selectedCell: {
    borderWidth: 2,
    borderColor: 'yellow',
  },
  resetButton: {
    backgroundColor: '#4a4a4a',
    padding: 10,
    borderRadius: 5,
  },
  resetButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default CheckersBoard;
