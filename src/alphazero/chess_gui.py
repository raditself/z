
import pygame
import sys
from chess_logic import ChessGame
from chess_ai import ChessAI
from game_analysis import GameAnalysis
import chess
import time
import tkinter as tk
from tkinter import filedialog
from visualization import visualize_move_explanations, plot_move_probabilities

# Initialize Pygame
pygame.init()

# Set display dimensions
WIDTH, HEIGHT = 640, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set title
pygame.display.set_caption("Chess AI with Explanations")

# Define board dimensions
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Define colors
WHITE = (255, 255, 255)
BLACK = (100, 100, 100)
HIGHLIGHT = (255, 255, 0)
BUTTON_COLOR = (150, 150, 150)
BUTTON_TEXT_COLOR = (0, 0, 0)
CLOCK_COLOR = (200, 200, 200)

# Initialize the chess game, AI, and game analysis
chess_game = ChessGame(initial_time=600)  # 10 minutes
chess_ai = ChessAI(depth=3)
game_analysis = GameAnalysis(chess_ai)

# Load piece images
piece_images = {}
for piece in ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k']:
    piece_images[piece] = pygame.image.load(f'chess_pieces/{piece}.png')
    piece_images[piece] = pygame.transform.scale(piece_images[piece], (SQUARE_SIZE, SQUARE_SIZE))

# Initialize font
pygame.font.init()
font = pygame.font.Font(None, 36)

# Add a flag for showing explanations
show_explanation = False

def draw_board():
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            piece = chess_game.get_piece_at(chess.square(col, 7-row))
            if piece != '.':
                screen.blit(piece_images[piece], (col * SQUARE_SIZE, row * SQUARE_SIZE))

def highlight_square(square):
    col, row = square % 8, 7 - (square // 8)
    pygame.draw.rect(screen, HIGHLIGHT, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

def draw_button(text, x, y, width, height):
    pygame.draw.rect(screen, BUTTON_COLOR, (x, y, width, height))
    text_surface = font.render(text, True, BUTTON_TEXT_COLOR)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surface, text_rect)

def draw_clock():
    white_time = chess_game.format_time(chess_game.white_time)
    black_time = chess_game.format_time(chess_game.black_time)
    
    pygame.draw.rect(screen, CLOCK_COLOR, (0, HEIGHT - 160, WIDTH, 40))
    white_text = font.render(f"White: {white_time}", True, BLACK)
    black_text = font.render(f"Black: {black_time}", True, BLACK)
    
    screen.blit(white_text, (20, HEIGHT - 155))
    screen.blit(black_text, (WIDTH - 150, HEIGHT - 155))

def update_display():
    draw_board()
    draw_clock()
    draw_button("New Game", 20, HEIGHT - 100, 120, 40)
    draw_button("AI Move", WIDTH - 140, HEIGHT - 100, 120, 40)
    draw_button("Load PGN", WIDTH // 2 - 60, HEIGHT - 100, 120, 40)
    draw_button("Toggle Explanations", WIDTH // 2 - 90, HEIGHT - 50, 180, 40)
    pygame.display.update()

def get_square_from_pos(pos):
    x, y = pos
    col = x // SQUARE_SIZE
    row = 7 - (y // SQUARE_SIZE)
    return chess.square(col, row)

def load_pgn():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("PGN files", "*.pgn")])
    if file_path:
        with open(file_path, 'r') as pgn_file:
            pgn_content = pgn_file.read()
            game = game_analysis.load_pgn(pgn_content)

def make_ai_move():
    ai_move = chess_ai.get_move(chess_game.board)
    chess_game.make_move(ai_move)
    if show_explanation:
        explanations = chess_ai.get_move_explanations()
        visualize_move_explanations(explanations)
        plot_move_probabilities(chess_ai.move_evaluations)

def main():
    global show_explanation
    selected_square = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if y < HEIGHT - 160:  # Click on the board
                    square = get_square_from_pos((x, y))
                    if selected_square is None:
                        selected_square = square
                    else:
                        move = chess.Move(selected_square, square)
                        if move in chess_game.board.legal_moves:
                            chess_game.make_move(move)
                            selected_square = None
                        else:
                            selected_square = square
                else:  # Click on buttons
                    if 20 <= x <= 140 and HEIGHT - 100 <= y <= HEIGHT - 60:
                        chess_game = ChessGame(initial_time=600)
                    elif WIDTH - 140 <= x <= WIDTH - 20 and HEIGHT - 100 <= y <= HEIGHT - 60:
                        make_ai_move()
                    elif WIDTH // 2 - 60 <= x <= WIDTH // 2 + 60 and HEIGHT - 100 <= y <= HEIGHT - 60:
                        load_pgn()
                    elif WIDTH // 2 - 90 <= x <= WIDTH // 2 + 90 and HEIGHT - 50 <= y <= HEIGHT - 10:
                        show_explanation = not show_explanation
                        print(f"Explanations {'enabled' if show_explanation else 'disabled'}")

        update_display()
        if selected_square is not None:
            highlight_square(selected_square)
        pygame.display.flip()

        if chess_game.is_game_over():
            print(f"Game Over! Result: {chess_game.get_result()}")
            chess_game = ChessGame(initial_time=600)  # Start a new game
            selected_square = None

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
