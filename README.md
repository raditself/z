
# Advanced Chess and Checkers AI Project

This project implements advanced AI systems for playing chess and checkers, including a mobile application for playing against the AI.

## Project Structure

- `src/`: Contains the main source code for the AI implementations
  - `alphazero/`: AlphaZero implementation for chess
  - `games/`: Game logic for chess and checkers
  - `reinforcement_learning/`: Reinforcement learning implementations
  - `web/`: Web API for AI integration
- `mobile_app/`: React Native mobile application for playing against the AI
- `tests/`: Unit and integration tests
- `docs/`: Project documentation

## Features

- AlphaZero-based chess AI
- Checkers AI with reinforcement learning
- Opening book system with support for multiple chess variants
- Endgame tablebases for perfect endgame play
- Distributed training system for faster learning
- Mobile application for playing against the AI
- Web API for AI move generation

## Getting Started

### Setting up the AI and Web API

1. Clone the repository:
   ```
   git clone https://github.com/raditself/z.git
   cd z
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the web API:
   ```
   python src/web/api.py
   ```

### Running the Mobile App

Please refer to the [mobile app README](./mobile_app/README.md) for detailed instructions on setting up and running the mobile application.

## Documentation

For more detailed information about the project, please refer to the [full documentation](./docs/full_documentation.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
