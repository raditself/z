
import numpy as np
import copy
import chess
from .chess_timer import ChessTimer
from .chess_variants import setup_chess960_board

class ChessGame:
    def __init__(self, initial_time=600, increment=10, variant='standard'):  # 10 minutes + 10 seconds increment
        self.variant = variant
        self.board = self.init_board()
        self.current_player = 1  # 1 for white, -1 for black
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_target = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.move_history = []
        self.action_size = 64 * 64  # All possible moves from any square to any square
        self.timer = ChessTimer(initial_time, increment)
        if self.variant == 'crazyhouse':
            self.piece_reserve = {1: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, -1: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}}

    def to_chess_board(self):
        chess_board = chess.Board(fen=None)
        chess_board.clear()
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j][0]
                if piece != 0:
                    color = chess.WHITE if piece > 0 else chess.BLACK
                    piece_type = abs(piece)
                    chess_piece = None
                    if piece_type == 1:
                        chess_piece = chess.PAWN
                    elif piece_type == 2:
                        chess_piece = chess.KNIGHT
                    elif piece_type == 3:
                        chess_piece = chess.BISHOP
                    elif piece_type == 4:
                        chess_piece = chess.ROOK
                    elif piece_type == 5:
                        chess_piece = chess.QUEEN
                    elif piece_type == 6:
                        chess_piece = chess.KING
                    chess_board.set_piece_at(chess.square(j, 7-i), chess.Piece(chess_piece, color))
        
        chess_board.turn = chess.WHITE if self.current_player == 1 else chess.BLACK
        chess_board.castling_rights = chess.CASTLING_RIGHTS[(self.castling_rights['K'], self.castling_rights['Q'], self.castling_rights['k'], self.castling_rights['q'])]
        chess_board.ep_square = chess.parse_square(self.en_passant_target) if self.en_passant_target else None
        chess_board.halfmove_clock = self.halfmove_clock
        chess_board.fullmove_number = self.fullmove_number
        
        return chess_board

    def clone(self):
        new_game = ChessGame(variant=self.variant)
        new_game.board = copy.deepcopy(self.board)
        new_game.current_player = self.current_player
        new_game.castling_rights = self.castling_rights.copy()
        new_game.en_passant_target = self.en_passant_target
        new_game.halfmove_clock = self.halfmove_clock
        new_game.fullmove_number = self.fullmove_number
        if self.variant == 'crazyhouse':
            new_game.piece_reserve = copy.deepcopy(self.piece_reserve)
        return new_game

    def evaluate(self):
        # Piece values
        piece_values = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 20000}
        
        # Initialize score
        score = 0
        
        # Evaluate material and position
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j][0]
                if piece != 0:
                    # Add material value
                    score += piece_values[abs(piece)] * np.sign(piece)
                    
                    # Add position value (simplified)
                    if abs(piece) == 1:  # Pawn
                        score += (i - 3.5) * 10 * np.sign(piece)  # Encourage pawn advancement
                    elif abs(piece) == 2:  # Knight
                        score += (3.5 - abs(i - 3.5) - abs(j - 3.5)) * 5 * np.sign(piece)  # Knights are better in the center
                    elif abs(piece) == 3:  # Bishop
                        score += (7 - abs(i - j)) * 5 * np.sign(piece)  # Bishops are better on long diagonals
                    elif abs(piece) == 4:  # Rook
                        if i == 7 or i == 0:
                            score += 20 * np.sign(piece)  # Rooks are better on open files
                    elif abs(piece) == 6:  # King
                        if self.fullmove_number <= 40:  # Early/mid game
                            score += (abs(j - 3.5) - 3.5) * 10 * np.sign(piece)  # King safety: keep the king on the side
                        else:  # Late game
                            score += (3.5 - abs(i - 3.5) - abs(j - 3.5)) * 10 * np.sign(piece)  # King should be active in the endgame
        
        # Add score for pieces in reserve (for crazyhouse variant)
        if self.variant == 'crazyhouse':
            for color in [1, -1]:
                for piece, count in self.piece_reserve[color].items():
                    score += piece_values[piece] * count * color
        
        return score if self.current_player == 1 else -score

    def init_board(self):
        if self.variant in ['standard', 'crazyhouse']:
            # Initialize the chess board
            board = np.zeros((8, 8, 6), dtype=np.int8)
            # Set up initial positions (simplified for this example)
            # 0: Pawn, 1: Rook, 2: Knight, 3: Bishop, 4: Queen, 5: King
            board[1] = board[6] = [1, 0, 0, 0, 0, 0]  # Pawns
            board[0] = board[7] = [0, 1, 1, 1, 1, 1]  # Other pieces
            board[7] *= -1
            board[6] *= -1
        elif self.variant == 'chess960':
            board = np.array(setup_chess960_board())
        else:
            raise ValueError(f"Unsupported chess variant: {self.variant}")
        return board

    def _get_drop_moves(self):
        drop_moves = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j][0] == 0:  # Empty square
                    for piece_type, count in self.piece_reserve[self.current_player].items():
                        if count > 0:
                            if piece_type == 1:  # Pawn
                                if i != 0 and i != 7:  # Pawns can't be dropped on the first or last rank
                                    drop_moves.append((None, (i, j), piece_type))
                            else:
                                drop_moves.append((None, (i, j), piece_type))
        return drop_moves

    def get_legal_moves(self):
        legal_moves = []
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece[0] * self.current_player > 0:  # Check if it's the current player's piece
                    if abs(piece[0]) == 1:  # Pawn
                        legal_moves.extend(self._get_pawn_moves(i, j))
                    elif abs(piece[0]) == 2:  # Knight
                        legal_moves.extend(self._get_knight_moves(i, j))
                    elif abs(piece[0]) == 3:  # Bishop
                        legal_moves.extend(self._get_bishop_moves(i, j))
                    elif abs(piece[0]) == 4:  # Rook
                        legal_moves.extend(self._get_rook_moves(i, j))
                    elif abs(piece[0]) == 5:  # Queen
                        legal_moves.extend(self._get_queen_moves(i, j))
                    elif abs(piece[0]) == 6:  # King
                        legal_moves.extend(self._get_king_moves(i, j))
        
        if self.variant == 'crazyhouse':
            legal_moves.extend(self._get_drop_moves())
        
        return legal_moves

    def _get_sliding_moves(self, row, col, directions):
        moves = []
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + i*dr, col + i*dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target_piece = self.board[new_row][new_col][0]
                    if target_piece * self.current_player <= 0:  # Empty or opponent's piece
                        moves.append((row, col, new_row, new_col))
                        if target_piece != 0:  # Stop if we hit a piece
                            break
                    else:
                        break  # Stop if we hit our own piece
                else:
                    break  # Stop if we're off the board
        return moves

    def _get_bishop_moves(self, row, col):
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return self._get_sliding_moves(row, col, directions)

    def _get_rook_moves(self, row, col):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return self._get_sliding_moves(row, col, directions)

    def _get_queen_moves(self, row, col):
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]
        return self._get_sliding_moves(row, col, directions)

    def _get_king_moves(self, row, col):
        moves = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if self.board[new_row][new_col][0] * self.current_player <= 0:  # Empty or opponent's piece
                    moves.append((row, col, new_row, new_col))
        
        # Castling
        if self.current_player == 1:
            if self.castling_rights['K'] and self._can_castle_kingside():
                moves.append((row, col, row, col + 2))
            if self.castling_rights['Q'] and self._can_castle_queenside():
                moves.append((row, col, row, col - 2))
        else:
            if self.castling_rights['k'] and self._can_castle_kingside():
                moves.append((row, col, row, col + 2))
            if self.castling_rights['q'] and self._can_castle_queenside():
                moves.append((row, col, row, col - 2))
        
        return moves

    def _can_castle_kingside(self):
        row = 0 if self.current_player == 1 else 7
        return (self.board[row][5].sum() == 0 and
                self.board[row][6].sum() == 0 and
                not self._is_square_attacked(row, 4) and
                not self._is_square_attacked(row, 5) and
                not self._is_square_attacked(row, 6))

    def _can_castle_queenside(self):
        row = 0 if self.current_player == 1 else 7
        return (self.board[row][1].sum() == 0 and
                self.board[row][2].sum() == 0 and
                self.board[row][3].sum() == 0 and
                not self._is_square_attacked(row, 2) and
                not self._is_square_attacked(row, 3) and
                not self._is_square_attacked(row, 4))

    def _is_square_attacked(self, row, col):
        # This is a simplified version. In a real implementation, you'd need to check all possible attacking moves.
        return any(move[2] == row and move[3] == col for move in self.get_legal_moves())

    def _get_knight_moves(self, row, col):
        moves = []
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if self.board[new_row][new_col][0] * self.current_player <= 0:  # Empty or opponent's piece
                    moves.append((row, col, new_row, new_col))
        return moves

    def _get_pawn_moves(self, row, col):
        moves = []
        direction = self.current_player
        start_row = 1 if direction == 1 else 6

        # Move forward
        if 0 <= row + direction < 8 and self.board[row + direction][col].sum() == 0:
            moves.append((row, col, row + direction, col))
            # Double move from starting position
            if row == start_row and self.board[row + 2*direction][col].sum() == 0:
                moves.append((row, col, row + 2*direction, col))

        # Capture diagonally
        for dc in [-1, 1]:
            if 0 <= row + direction < 8 and 0 <= col + dc < 8:
                if self.board[row + direction][col + dc][0] * self.current_player < 0:
                    moves.append((row, col, row + direction, col + dc))

        # En passant
        if self.en_passant_target:
            ep_row, ep_col = self.en_passant_target
            if row == (3 if direction == 1 else 4) and abs(col - ep_col) == 1:
                moves.append((row, col, ep_row, ep_col))

        return moves

    def make_move(self, move):
        try:
            from_row, from_col, to_row, to_col = move
            if not (0 <= from_row < 8 and 0 <= from_col < 8 and 0 <= to_row < 8 and 0 <= to_col < 8):
                raise ValueError("Invalid move: out of board boundaries")

            moving_piece = self.board[from_row][from_col]
            if moving_piece[0] * self.current_player <= 0:
                raise ValueError("Invalid move: no piece or opponent's piece at the starting position")

            if (from_row, from_col, to_row, to_col) not in self.get_legal_moves():
                raise ValueError("Invalid move: not a legal move")

            captured_piece = self.board[to_row][to_col]

            # Handle castling
            if abs(moving_piece[5]) == 1 and abs(to_col - from_col) == 2:
                if to_col > from_col:  # Kingside castling
                    self.board[to_row][to_col-1] = self.board[to_row][7]
                    self.board[to_row][7] = 0
                else:  # Queenside castling
                    self.board[to_row][to_col+1] = self.board[to_row][0]
                    self.board[to_row][0] = 0

            # Handle en passant
            if abs(moving_piece[0]) == 1 and (to_row, to_col) == self.en_passant_target:
                self.board[from_row][to_col] = 0  # Remove the captured pawn

            # Move the piece
            self.board[to_row][to_col] = moving_piece
            self.board[from_row][from_col] = 0

            # Update castling rights
            if abs(moving_piece[5]) == 1:  # King moved
                if self.current_player == 1:
                    self.castling_rights['K'] = self.castling_rights['Q'] = False
                else:
                    self.castling_rights['k'] = self.castling_rights['q'] = False
            elif abs(moving_piece[1]) == 1:  # Rook moved
                if from_row == 0 and from_col == 0: self.castling_rights['Q'] = False
                elif from_row == 0 and from_col == 7: self.castling_rights['K'] = False
                elif from_row == 7 and from_col == 0: self.castling_rights['q'] = False
                elif from_row == 7 and from_col == 7: self.castling_rights['k'] = False

            # Set en passant target
            if abs(moving_piece[0]) == 1 and abs(to_row - from_row) == 2:
                self.en_passant_target = ((from_row + to_row) // 2, from_col)
            else:
                self.en_passant_target = None

            # Update halfmove clock
            if abs(moving_piece[0]) == 1 or captured_piece.sum() != 0:
                self.halfmove_clock = 0
            else:
                self.halfmove_clock += 1

            # Update fullmove number
            if self.current_player == -1:
                self.fullmove_number += 1

            # Record the move in move history using algebraic notation
            self.move_history.append(self.move_to_algebraic(move))

            # Switch player
            self.current_player *= -1

        except ValueError as e:
            print(f"Error: {str(e)}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            raise

    def move_to_algebraic(self, move):
        from_row, from_col, to_row, to_col = move
        piece = self.board[from_row][from_col]
        piece_symbol = self.piece_to_symbol(piece[0])
        from_square = self.coord_to_algebraic(from_row, from_col)
        to_square = self.coord_to_algebraic(to_row, to_col)
        capture = 'x' if self.board[to_row][to_col].sum() != 0 else ''
        return f"{piece_symbol}{from_square}{capture}{to_square}"

    def piece_to_symbol(self, piece):
        symbols = {1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K'}
        return symbols[abs(piece)] if abs(piece) in symbols else ''

    def coord_to_algebraic(self, row, col):
        return f"{chr(97 + col)}{8 - row}"

    def algebraic_to_move(self, algebraic):
        if algebraic == "O-O":  # Kingside castling
            return (7, 4, 7, 6) if self.current_player == 1 else (0, 4, 0, 6)
        elif algebraic == "O-O-O":  # Queenside castling
            return (7, 4, 7, 2) if self.current_player == 1 else (0, 4, 0, 2)

        piece_symbol = algebraic[0] if algebraic[0].isupper() else 'P'
        to_square = algebraic[-2:]
        to_col = ord(to_square[0]) - 97
        to_row = 8 - int(to_square[1])

        if piece_symbol == 'P':
            from_col = to_col
            from_row = to_row + (1 if self.current_player == 1 else -1)
        else:
            # Find the piece that can make this move
            for row in range(8):
                for col in range(8):
                    piece = self.board[row][col]
                    if piece[0] * self.current_player > 0 and self.piece_to_symbol(piece[0]) == piece_symbol:
                        if (row, col, to_row, to_col) in self.get_legal_moves():
                            from_row, from_col = row, col
                            break
                else:
                    continue
                break

        return (from_row, from_col, to_row, to_col)

    def get_legal_moves_for_piece(self, row, col):
        """
        Returns a list of legal moves for the piece at the given position.
        """
        piece = self.board[row][col]
        if piece[0] * self.current_player <= 0:  # Empty square or opponent's piece
            return []

        all_legal_moves = self.get_legal_moves()
        return [(r, c) for (fr, fc, r, c) in all_legal_moves if fr == row and fc == col]

    def highlight_legal_moves(self, row, col):
        """
        Returns a 2D list representing the board with highlighted legal moves.
        0: Empty square
        1: Piece
        2: Highlighted square (legal move)
        """
        highlighted_board = [[1 if self.board[r][c].sum() != 0 else 0 for c in range(8)] for r in range(8)]
        legal_moves = self.get_legal_moves_for_piece(row, col)
        for r, c in legal_moves:
            highlighted_board[r][c] = 2
        return highlighted_board

    def is_game_over(self):
        if self.is_checkmate() or self.is_stalemate() or self.is_draw_by_insufficient_material() or self.is_draw_by_repetition() or self.is_draw_by_fifty_move_rule():
            return True
        return self.timer.is_flag_fallen('white') or self.timer.is_flag_fallen('black')

    def get_winner(self):
        if self.timer.is_flag_fallen('white'):
            return -1  # Black wins
        elif self.timer.is_flag_fallen('black'):
            return 1  # White wins
        if self.is_checkmate():
            return -self.current_player  # The other player wins
        return 0  # Draw or game not over

    def is_checkmate(self):
        if self.is_in_check():
            return len(self.get_legal_moves()) == 0
        return False

    def is_stalemate(self):
        if not self.is_in_check():
            return len(self.get_legal_moves()) == 0
        return False

    def is_draw_by_insufficient_material(self):
        pieces = [piece for row in self.board for piece in row if piece[0] != 0]
        if len(pieces) == 2:  # Only kings left
            return True
        if len(pieces) == 3 and any(abs(piece[0]) in [2, 3] for piece in pieces):  # King and bishop/knight vs king
            return True
        return False

    def is_draw_by_repetition(self):
        # This would require keeping a history of positions, which we don't currently have
        # For simplicity, we'll return False, but in a full implementation, you'd check for threefold repetition
        return False

    def is_draw_by_fifty_move_rule(self):
        return self.halfmove_clock >= 100  # 50 full moves (100 half-moves)

    def get_state(self):
        return self.board.copy(), self.current_player

    def get_remaining_time(self, color):
        return self.timer.get_time('white' if color == 1 else 'black')

    def is_in_check(self):
        # Find the king's position
        king_pos = None
        for i in range(8):
            for j in range(8):
                if self.board[i][j][0] * self.current_player == 6:  # 6 represents the king
                    king_pos = (i, j)
                    break
            if king_pos:
                break

        if not king_pos:
            return False  # This should not happen in a valid chess position

        # Check if any opponent's piece can attack the king
        for i in range(8):
            for j in range(8):
                if self.board[i][j][0] * self.current_player < 0:  # Opponent's piece
                    if (king_pos[0], king_pos[1]) in self.get_legal_moves_for_piece(i, j):
                        return True

        return False

    def get_result(self):
        if self.is_checkmate():
            return "0-1" if self.current_player == 1 else "1-0"
        elif self.is_stalemate() or self.is_draw_by_insufficient_material() or self.is_draw_by_repetition() or self.is_draw_by_fifty_move_rule():
            return "1/2-1/2"
        elif self.timer.is_flag_fallen('white'):
            return "0-1"
        elif self.timer.is_flag_fallen('black'):
            return "1-0"
        return None
