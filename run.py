from src.alphazero.game import ChessGame
from src.alphazero.ai import ChessAI
from src.alphazero.game_analysis import analyze_game, suggest_improvements

def main():
    # Initialize the game
    game = ChessGame(initial_time=600, increment=10, variant='standard')  # 10 minutes + 10 seconds increment
    ai = ChessAI(difficulty='medium')

    # Main game loop
    while not game.is_game_over():
        print(game.board)
        print(f"Current player: {'White' if game.current_player == 1 else 'Black'}")
        print(f"Remaining time - White: {game.get_remaining_time(1):.1f}s, Black: {game.get_remaining_time(-1):.1f}s")

        if game.current_player == 1:  # Human player (White)
            move = input("Enter your move (e.g., e2e4): ")
            move = game.algebraic_to_move(move)
        else:  # AI player (Black)
            move = ai.get_move(game)

        # Highlight legal moves for the selected piece
        if game.current_player == 1:
            from_row, from_col, _, _ = move
            highlighted_board = game.highlight_legal_moves(from_row, from_col)
            print("Highlighted legal moves:")
            print(highlighted_board)

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

if __name__ == "__main__":
    main()
