
import pygame
import sys
from src.alphazero.chess_ai import ChessAI
from src.alphazero.checkers_ai import CheckersAI
from src.games.chess import ChessGame
from src.games.checkers import CheckersGame

class GameGUI:
    def __init__(self, game_type='chess', ai_model_path=None):
        pygame.init()
        self.game_type = game_type
        self.width = 800
        self.height = 600
        self.board_size = 400
        self.square_size = self.board_size // 8
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Play {game_type.capitalize()} against AI")

        if game_type == 'chess':
            self.game = ChessGame()
            self.ai = ChessAI(ai_model_path)
        elif game_type == 'checkers':
            self.game = CheckersGame()
            self.ai = CheckersAI(ai_model_path)
        else:
            raise ValueError("Invalid game type. Choose 'chess' or 'checkers'.")

        self.board = self.game.get_initial_board()
        self.selected_piece = None
        self.font = pygame.font.Font(None, 36)

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = (255, 206, 158) if (row + col) % 2 == 0 else (209, 139, 71)
                pygame.draw.rect(self.screen, color, (col * self.square_size, row * self.square_size, self.square_size, self.square_size))

    def draw_pieces(self):
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != 0:
                    if self.game_type == 'chess':
                        color = (255, 255, 255) if piece > 0 else (0, 0, 0)
                        pygame.draw.circle(self.screen, color, 
                                           (col * self.square_size + self.square_size // 2, 
                                            row * self.square_size + self.square_size // 2), 
                                           self.square_size // 2 - 5)
                        text = self.font.render(self.get_chess_piece_symbol(piece), True, (255 - color[0], 255 - color[1], 255 - color[2]))
                        text_rect = text.get_rect(center=(col * self.square_size + self.square_size // 2, 
                                                          row * self.square_size + self.square_size // 2))
                        self.screen.blit(text, text_rect)
                    elif self.game_type == 'checkers':
                        color = (255, 0, 0) if piece > 0 else (0, 0, 0)
                        pygame.draw.circle(self.screen, color, 
                                           (col * self.square_size + self.square_size // 2, 
                                            row * self.square_size + self.square_size // 2), 
                                           self.square_size // 2 - 5)
                        if abs(piece) == 2:  # King piece
                            pygame.draw.circle(self.screen, (255, 255, 0), 
                                               (col * self.square_size + self.square_size // 2, 
                                                row * self.square_size + self.square_size // 2), 
                                               self.square_size // 4)

    def get_chess_piece_symbol(self, piece):
        symbols = {1: '♙', 2: '♘', 3: '♗', 4: '♖', 5: '♕', 6: '♔',
                   -1: '♟', -2: '♞', -3: '♝', -4: '♜', -5: '♛', -6: '♚'}
        return symbols.get(piece, '')

    def draw_game_info(self):
        info_text = f"Game: {self.game_type.capitalize()}"
        text = self.font.render(info_text, True, (0, 0, 0))
        self.screen.blit(text, (self.board_size + 20, 20))

        current_player = "White" if self.game.get_current_player(self.board) == 1 else "Black"
        player_text = f"Current player: {current_player}"
        text = self.font.render(player_text, True, (0, 0, 0))
        self.screen.blit(text, (self.board_size + 20, 60))

        if self.game.is_game_over(self.board):
            winner = self.game.get_winner(self.board)
            if winner == 1:
                result_text = "You win!"
            elif winner == -1:
                result_text = "AI wins!"
            else:
                result_text = "It's a draw!"
            text = self.font.render(result_text, True, (0, 0, 0))
            self.screen.blit(text, (self.board_size + 20, 100))

    def handle_click(self, pos):
        if self.game.is_game_over(self.board):
            return

        col = pos[0] // self.square_size
        row = pos[1] // self.square_size

        if self.selected_piece is None:
            self.selected_piece = (row, col)
        else:
            move = (self.selected_piece[0], self.selected_piece[1], row, col)
            if self.game.is_valid_move(self.board, move):
                self.board = self.game.make_move(self.board, move)
                self.selected_piece = None

                if not self.game.is_game_over(self.board):
                    ai_move = self.ai.get_move(self.board)
                    self.board = self.game.make_move(self.board, ai_move)
            else:
                self.selected_piece = None

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)

            self.screen.fill((255, 255, 255))
            self.draw_board()
            self.draw_pieces()
            self.draw_game_info()
            pygame.display.flip()

# Usage example:
# chess_gui = GameGUI('chess', 'path_to_chess_model.h5')
# chess_gui.run()

# checkers_gui = GameGUI('checkers', 'path_to_checkers_model.h5')
# checkers_gui.run()
