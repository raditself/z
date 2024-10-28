import pygame
from checkers import CheckersGame

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 800
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)

# Board dimensions
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

def draw_board(game):
    SCREEN.fill(WHITE)
    for row in range(ROWS):
        for col in range(COLS):
            if (row + col) % 2 == 0:
                pygame.draw.rect(SCREEN, GRAY, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            
            piece = game.board[row][col]
            if piece != 0:
                color = WHITE if piece == 1 else BLACK
                pygame.draw.circle(SCREEN, color, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 10)

def get_row_col_from_mouse(pos):
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col

def main():
    game = CheckersGame()
    clock = pygame.time.Clock()
    selected_piece = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = get_row_col_from_mouse(pos)
                
                if selected_piece is None:
                    if game.board[row][col] == game.current_player:
                        selected_piece = (row, col)
                else:
                    start_row, start_col = selected_piece
                    move = (start_row, start_col, row, col)
                    if move in game.get_valid_moves(game.current_player):
                        game.make_move(move)
                    selected_piece = None

        draw_board(game)
        if selected_piece:
            row, col = selected_piece
            pygame.draw.circle(SCREEN, RED, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 5, 5)
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
