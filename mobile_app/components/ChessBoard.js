
import React, { useState, useEffect } from 'react';
import { View, StyleSheet, TouchableOpacity, Text, Button, Alert } from 'react-native';
import { Chess } from 'chess.js';
import { makeMove } from './ChessAPI';

const BOARD_SIZE = 8;
const SQUARE_SIZE = 40;

const ChessBoard = () => {
  const [game, setGame] = useState(new Chess());
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [isThinking, setIsThinking] = useState(false);

  const renderSquare = (row, col) => {
    const isBlack = (row + col) % 2 === 1;
    const square = String.fromCharCode(97 + col) + (8 - row);
    const piece = game.get(square);

    return (
      <TouchableOpacity
        key={`${row}-${col}`}
        style={[
          styles.square,
          isBlack ? styles.blackSquare : styles.whiteSquare,
          selectedSquare === square ? styles.selectedSquare : null,
        ]}
        onPress={() => handleSquarePress(square)}
        disabled={isThinking}
      >
        <Text style={styles.piece}>{piece ? pieceToUnicode(piece) : ''}</Text>
      </TouchableOpacity>
    );
  };

  const handleSquarePress = async (square) => {
    if (selectedSquare) {
      const move = game.move({
        from: selectedSquare,
        to: square,
        promotion: 'q', // Always promote to queen for simplicity
      });

      if (move) {
        setGame(new Chess(game.fen()));
        setSelectedSquare(null);
        setIsThinking(true);
        
        try {
          const result = await makeMove(game.fen(), move.san);
          if (result) {
            game.move(result.ai_move);
            setGame(new Chess(game.fen()));
            
            if (result.game_over) {
              Alert.alert('Game Over', `Result: ${result.result}`);
            }
          }
        } catch (error) {
          console.error('Error making AI move:', error);
          Alert.alert('Error', 'Failed to get AI move. Please try again.');
        } finally {
          setIsThinking(false);
        }
      } else {
        setSelectedSquare(square);
      }
    } else {
      setSelectedSquare(square);
    }
  };

  const resetGame = () => {
    setGame(new Chess());
    setSelectedSquare(null);
    setIsThinking(false);
  };

  const pieceToUnicode = (piece) => {
    const pieceUnicode = {
      'p': '♙', 'n': '♘', 'b': '♗', 'r': '♖', 'q': '♕', 'k': '♔',
      'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛', 'K': '♚'
    };
    return pieceUnicode[piece.type] || '';
  };

  return (
    <View style={styles.container}>
      <View style={styles.board}>
        {[...Array(BOARD_SIZE)].map((_, row) => (
          <View key={row} style={styles.row}>
            {[...Array(BOARD_SIZE)].map((_, col) => renderSquare(row, col))}
          </View>
        ))}
      </View>
      <Button title="Reset Game" onPress={resetGame} disabled={isThinking} />
      <Text style={styles.status}>
        {isThinking ? 'AI is thinking...' : `${game.turn() === 'w' ? 'White' : 'Black'} to move`}
      </Text>
      {game.game_over() && <Text style={styles.gameOver}>Game Over</Text>}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
  },
  board: {
    flexDirection: 'column',
    borderWidth: 2,
    borderColor: '#000',
  },
  row: {
    flexDirection: 'row',
  },
  square: {
    width: SQUARE_SIZE,
    height: SQUARE_SIZE,
    justifyContent: 'center',
    alignItems: 'center',
  },
  blackSquare: {
    backgroundColor: '#769656',
  },
  whiteSquare: {
    backgroundColor: '#eeeed2',
  },
  selectedSquare: {
    backgroundColor: '#baca44',
  },
  piece: {
    fontSize: 24,
  },
  status: {
    marginTop: 10,
    fontSize: 18,
  },
  gameOver: {
    marginTop: 10,
    fontSize: 24,
    fontWeight: 'bold',
    color: 'red',
  },
});

export default ChessBoard;
