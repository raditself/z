
# AI Board Games Mobile App

This mobile app allows users to play Chess and Checkers against an advanced AI opponent. The app is built using React Native and includes local AI implementations for both games.

## Prerequisites

- Node.js (v12 or later)
- npm (v6 or later)
- React Native CLI
- Xcode (for iOS development)
- Android Studio (for Android development)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/raditself/z.git
   cd z/mobile_app
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the Metro bundler:
   ```
   npx react-native start
   ```

4. Run the app on iOS:
   ```
   npx react-native run-ios
   ```

   Or on Android:
   ```
   npx react-native run-android
   ```

## Playing the Games

1. Launch the app on your device or emulator.
2. Choose between Chess and Checkers on the main screen.
3. Select a difficulty level (Easy, Medium, or Hard).
4. For Chess:
   - The chessboard will be displayed with white pieces at the bottom.
   - Tap a piece to select it, then tap a valid square to move the piece.
   - The AI will automatically make its move after you've made yours.
5. For Checkers:
   - The checkers board will be displayed with red pieces at the bottom.
   - Tap a piece to select it, then tap a valid square to move the piece.
   - The AI will automatically make its move after you've made yours.
6. Use the "Reset Game" button to start a new game at any time.

Enjoy playing against the AI!

## Features

- Two classic board games: Chess and Checkers
- Three difficulty levels for each game
- Local AI implementation for offline play
- Intuitive touch controls for piece movement
- Game state indicators (current player, AI thinking)
- Reset game functionality

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are correctly installed.
2. Check the console for any error messages.
3. Ensure you have the latest version of React Native and its dependencies.

For more detailed information on React Native development and troubleshooting, refer to the [React Native documentation](https://reactnative.dev/docs/environment-setup).

## Future Improvements

- Online multiplayer functionality
- More board games (e.g., Go, Othello)
- Customizable board themes
- Game analysis and move suggestions
- Integration with chess/checkers engines for higher difficulty levels

Feel free to contribute to the project by submitting pull requests or reporting issues on the GitHub repository.
