
import pygame
import sys
from chess_logic import ChessGame
from checkers import Checkers
from chess_ai import ChessAI
from checkers_ai import CheckersAI
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
    def __init__(self):
        self.game_type = None
        self.game = None
        self.ai = None
        self.game_analysis = None
        self.board_surface = pygame.Surface((WIDTH, WIDTH))
        self.show_menu()

    def show_menu(self):
        menu_options = ['Chess', 'Checkers']
        button_height = 50
        button_width = 200
        button_margin = 20
        total_height = len(menu_options) * (button_height + button_margin) - button_margin
        start_y = (HEIGHT - total_height) // 2

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    for i, option in enumerate(menu_options):
                        button_y = start_y + i * (button_height + button_margin)
                        button_rect = pygame.Rect((WIDTH - button_width) // 2, button_y, button_width, button_height)
                        if button_rect.collidepoint(mouse_pos):
                            self.start_game(option.lower())
                            return

            screen.fill(WHITE)
            for i, option in enumerate(menu_options):
                button_y = start_y + i * (button_height + button_margin)
                self.draw_button(option, (WIDTH - button_width) // 2, button_y, button_width, button_height)
            pygame.display.flip()

    def start_game(self, game_type):
        self.game_type = game_type
        if game_type == 'chess':
            self.game = ChessGame(initial_time=600)  # 10 minutes
            self.ai = ChessAI(depth=3)
            self.game_analysis = GameAnalysis(self.ai)
            self.load_chess_pieces()
        elif game_type == 'checkers':
            self.game = Checkers()
            self.ai = CheckersAI(depth=3)
            self.load_checkers_pieces()
        
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

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # Add event handling for game moves here

            self.update_display()
            pygame.display.flip()

if __name__ == "__main__":
    gui = GameGUI()
    gui.run()
