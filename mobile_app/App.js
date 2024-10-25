
import React, { useState } from 'react';
import { SafeAreaView, StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import ChessBoard from './components/ChessBoard';
import CheckersBoard from './components/CheckersBoard';

const GameSelection = ({ onSelectGame }) => {
  return (
    <View style={styles.gameSelection}>
      <TouchableOpacity style={styles.gameButton} onPress={() => onSelectGame('chess')}>
        <Text style={styles.gameButtonText}>Chess</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.gameButton} onPress={() => onSelectGame('checkers')}>
        <Text style={styles.gameButtonText}>Checkers</Text>
      </TouchableOpacity>
    </View>
  );
};

const DifficultySelection = ({ onSelectDifficulty }) => {
  return (
    <View style={styles.difficultySelection}>
      <Text style={styles.difficultyTitle}>Select Difficulty</Text>
      <TouchableOpacity style={styles.difficultyButton} onPress={() => onSelectDifficulty('easy')}>
        <Text style={styles.difficultyButtonText}>Easy</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.difficultyButton} onPress={() => onSelectDifficulty('medium')}>
        <Text style={styles.difficultyButtonText}>Medium</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.difficultyButton} onPress={() => onSelectDifficulty('hard')}>
        <Text style={styles.difficultyButtonText}>Hard</Text>
      </TouchableOpacity>
    </View>
  );
};

const App = () => {
  const [selectedGame, setSelectedGame] = useState(null);
  const [difficulty, setDifficulty] = useState(null);

  const renderContent = () => {
    if (!selectedGame) {
      return <GameSelection onSelectGame={setSelectedGame} />;
    }
    if (!difficulty) {
      return <DifficultySelection onSelectDifficulty={setDifficulty} />;
    }
    switch (selectedGame) {
      case 'chess':
        return <ChessBoard difficulty={difficulty} />;
      case 'checkers':
        return <CheckersBoard difficulty={difficulty} />;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>AI Board Games</Text>
      </View>
      {renderContent()}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f0f0f0',
  },
  header: {
    backgroundColor: '#4a4a4a',
    padding: 20,
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  gameSelection: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  gameButton: {
    backgroundColor: '#4a4a4a',
    padding: 15,
    margin: 10,
    borderRadius: 5,
  },
  gameButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  difficultySelection: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  difficultyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  difficultyButton: {
    backgroundColor: '#4a4a4a',
    padding: 15,
    margin: 10,
    borderRadius: 5,
    width: 200,
    alignItems: 'center',
  },
  difficultyButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});

export default App;
