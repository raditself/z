import pygame
import sys
from chess_logic import ChessGame
from checkers import Checkers
from chess_ai import ChessAI
from game_analysis import GameAnalysis
import chess
import time
import tkinter as tk
from tkinter import filedialog

# Initialize Pygame
pygame.init()

# Set display dimensions
WIDTH, HEIGHT = 640, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set title
pygame.display.set_caption("Board Game AI")

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

# Initialize font
pygame.font.init()
font = pygame.font.Font(None, 36)

class GameGUI:
    def __init__(self, game_type='chess'):
        self.game_type = game_type
        if game_type == 'chess':
            self.game = ChessGame(initial_time=600)  # 10 minutes
            self.ai = ChessAI(depth=3)
            self.game_analysis = GameAnalysis(self.ai)
            self.load_chess_pieces()
        elif game_type == 'checkers':
            self.game = Checkers()
            self.load_checkers_pieces()
        
        self.board_surface = pygame.Surface((WIDTH, WIDTH))
        self.update_board_surface()

    def load_chess_pieces(self):
        self.piece_images = {}
        for piece in ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k']:
            self.piece_images[piece] = pygame.image.load(f'chess_pieces/{piece}.png')
            self.piece_images[piece] = pygame.transform.scale(self.piece_images[piece], (SQUARE_SIZE, SQUARE_SIZE))

    def load_checkers_pieces(self):
        self.piece_images = {
            1: pygame.transform.scale(pygame.image.load('checkers_pieces/red.png'), (SQUARE_SIZE, SQUARE_SIZE)),
            2: pygame.transform.scale(pygame.image.load('checkers_pieces/black.png'), (SQUARE_SIZE, SQUARE_SIZE))
        }

    def update_board_surface(self):
        for row in range(ROWS):
            for col in range(COLS):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                pygame.draw.rect(self.board_surface, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                if self.game_type == 'chess':
                    piece = self.game.get_piece_at(chess.square(col, 7-row))
                    if piece != '.':
                        self.board_surface.blit(self.piece_images[piece], (col * SQUARE_SIZE, row * SQUARE_SIZE))
                elif self.game_type == 'checkers':
                    piece = self.game.board[row][col]
                    if piece != 0:
                        self.board_surface.blit(self.piece_images[piece], (col * SQUARE_SIZE, row * SQUARE_SIZE))

    def draw_board(self):
        screen.blit(self.board_surface, (0, 0))

    def highlight_square(self, square):
        if self.game_type == 'chess':
            col, row = square % 8, 7 - (square // 8)
        elif self.game_type == 'checkers':
            col, row = square[1], square[0]
        pygame.draw.rect(screen, HIGHLIGHT, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

    def draw_button(self, text, x, y, width, height):
        pygame.draw.rect(screen, BUTTON_COLOR, (x, y, width, height))
        text_surface = font.render(text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
        screen.blit(text_surface, text_rect)

    def draw_clock(self):
        if self.game_type == 'chess':
            white_time = self.game.format_time(self.game.white_time)
            black_time = self.game.format_time(self.game.black_time)
            
            pygame.draw.rect(screen, CLOCK_COLOR, (0, HEIGHT - 160, WIDTH, 40))
            white_text = font.render(f"White: {white_time}", True, BLACK)
            black_text = font.render(f"Black: {black_time}", True, BLACK)
            
            screen.blit(white_text, (20, HEIGHT - 155))
            screen.blit(black_text, (WIDTH - 150, HEIGHT - 155))

    def update_display(self):
        self.draw_board()
        if self.game_type == 'chess':
            self.draw_clock()
        self.draw_button("New Game", 20, HEIGHT - 100, 120, 40)
        self.draw_button("AI Move", WIDTH - 140, HEIGHT - 100, 120, 40)
        if self.game_type == 'chess':
            self.draw_button("Load PGN", WIDTH // 2 - 60, HEIGHT - 100, 120, 40)
        pygame.display.update()

    def get_square_from_pos(self, pos):
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        if self.game_type == 'chess':
            return chess.square(col, 7 - row)
        elif self.game_type == 'checkers':
            return (row, col)

    def load_pgn(self):
        if self.game_type == 'chess':
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(filetypes=[("PGN files", "*.pgn")])
            if file_path:
                with open(file_path, 'r') as pgn_file:
                    pgn_content = pgn_file.read()
                    game = self.game_analysis.load_pgn(pgn_content)
                    analysis = self.game_analysis.analyze_game(game)
                    critical_positions = self.game_analysis.get_critical_positions(analysis)
                    improvements = self.game_analysis.suggest_improvements(game, analysis)

    def run(self):
        selected_square = None
        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if pos[1] < HEIGHT - 120:
                        square = self.get_square_from_pos(pos)
                        if selected_square is None:
                            selected_square = square
                        else:
                            # Make move
                            if self.game_type == 'chess':
                                move = chess.Move(selected_square, square)
                                if move in self.game.board.legal_moves:
                                    self.game.make_move(move)
                                    self.update_board_surface()
                            elif self.game_type == 'checkers':
                                move = (selected_square[0], selected_square[1], square[0], square[1])
                                if move in self.game.get_valid_moves():
                                    self.game.make_move(move)
                                    self.update_board_surface()
                            selected_square = None
                    else:
                        if 20 <= pos[0] <= 140 and HEIGHT - 100 <= pos[1] <= HEIGHT - 60:
                            # New Game button
                            if self.game_type == 'chess':
                                self.game = ChessGame(initial_time=600)
                            elif self.game_type == 'checkers':
                                self.game = Checkers()
                            self.update_board_surface()
                        elif WIDTH - 140 <= pos[0] <= WIDTH - 20 and HEIGHT - 100 <= pos[1] <= HEIGHT - 60:
                            # AI Move button
                            if self.game_type == 'chess':
                                ai_move = self.ai.get_best_move(self.game.board)
                                self.game.make_move(ai_move)
                                self.update_board_surface()
                            elif self.game_type == 'checkers':
                                # Implement Checkers AI move here
                                pass
                        elif self.game_type == 'chess' and WIDTH // 2 - 60 <= pos[0] <= WIDTH // 2 + 60 and HEIGHT - 100 <= pos[1] <= HEIGHT - 60:
                            # Load PGN button (Chess only)
                            self.load_pgn()

            self.update_display()
            if selected_square is not None:
                self.highlight_square(selected_square)
            pygame.display.flip()
            clock.tick(60)  # Limit to 60 FPS

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game_gui = GameGUI(game_type='chess')  # or 'checkers'
    game_gui.run()
