
# Chess AI Mobile App

This mobile app allows users to play chess against an advanced AI opponent. The app is built using React Native and communicates with a backend API for AI moves.

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

## Running the Backend API

The mobile app requires a backend API to be running for AI moves. To start the API:

1. Navigate to the project root directory:
   ```
   cd /path/to/z
   ```

2. Install Python dependencies (if not already installed):
   ```
   pip install -r requirements.txt
   ```

3. Start the API server:
   ```
   python src/web/api.py
   ```

The API should now be running on `http://localhost:8000`.

## Playing the Game

1. Launch the app on your device or emulator.
2. The chessboard will be displayed with white pieces at the bottom.
3. Tap a piece to select it, then tap a valid square to move the piece.
4. The AI will automatically make its move after you've made yours.
5. Use the "Reset Game" button to start a new game at any time.

Enjoy playing against the AI!

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are correctly installed.
2. Ensure the backend API is running and accessible.
3. Check the console for any error messages.

For more detailed information on React Native development and troubleshooting, refer to the [React Native documentation](https://reactnative.dev/docs/environment-setup).
