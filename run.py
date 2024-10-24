import argparse
from src.alphazero.game import ChessGame
from src.alphazero.ai import ChessAI
from src.alphazero.model import ChessModel
from src.alphazero.train import TrainAlphaZero
from src.alphazero.game_analysis import analyze_game, suggest_improvements

def train_alphazero(args):
    game = ChessGame()
    model = ChessModel()
    trainer = TrainAlphaZero(game, model, args)
    trainer.learn()

def play_against_ai(args):
    game = ChessGame(initial_time=600, increment=10, variant='standard')
    ai = ChessAI(difficulty=args.difficulty)

    while not game.is_game_over():
        print(game.board)
        print(f"Current player: {'White' if game.current_player == 1 else 'Black'}")

        if game.current_player == 1:  # Human player (White)
            move = input("Enter your move (e.g., e2e4): ")
            move = game.algebraic_to_move(move)
        else:  # AI player (Black)
            move = ai.get_move(game)

        game.make_move(move)

    print("Game Over!")
    print(f"Winner: {'White' if game.get_winner() == 1 else 'Black' if game.get_winner() == -1 else 'Draw'}")

    # Analyze the game
    analysis = analyze_game(game.move_history)
    improvements = suggest_improvements(analysis)

    print("\nGame Analysis:")
    for pos in analysis:
        print(f"Move {pos['move_number']}: Evaluation: {pos['evaluation']}")

    print("\nSuggested Improvements:")
    for suggestion in improvements:
        print(f"Move {suggestion['move_number']}: {suggestion['suggestion']}")

def main():
    parser = argparse.ArgumentParser(description="AlphaZero Chess")
    parser.add_argument("mode", choices=["train", "play"], help="Mode: train AlphaZero or play against AI")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium", help="AI difficulty when playing")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--episodes", type=int, default=100, help="Number of self-play episodes per iteration")
    parser.add_argument("--mcts_simulations", type=int, default=100, help="Number of MCTS simulations per move")
    args = parser.parse_args()

    if args.mode == "train":
        train_alphazero(args)
    elif args.mode == "play":
        play_against_ai(args)

if __name__ == "__main__":
    main()
