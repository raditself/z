
import numpy as np

import copy

class ChessGame:
    def __init__(self):
        self.board = self.init_board()
        self.current_player = 1  # 1 for white, -1 for black
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_target = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.action_size = 64 * 64  # All possible moves from any square to any square

    def clone(self):
        new_game = ChessGame()
        new_game.board = copy.deepcopy(self.board)
        new_game.current_player = self.current_player
        new_game.castling_rights = self.castling_rights.copy()
        new_game.en_passant_target = self.en_passant_target
        new_game.halfmove_clock = self.halfmove_clock
        new_game.fullmove_number = self.fullmove_number
        return new_game

    def init_board(self):
        # Initialize the chess board
        board = np.zeros((8, 8, 6), dtype=np.int8)
        # Set up initial positions (simplified for this example)
        # 0: Pawn, 1: Rook, 2: Knight, 3: Bishop, 4: Queen, 5: King
        board[1] = board[6] = [1, 0, 0, 0, 0, 0]  # Pawns
        board[0] = board[7] = [0, 1, 1, 1, 1, 1]  # Other pieces
        board[7] *= -1
        board[6] *= -1
        return board

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

            # Switch player
            self.current_player *= -1

        except ValueError as e:
            print(f"Error: {str(e)}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            raise

    def is_game_over(self):
        return self.is_checkmate() or self.is_stalemate() or self.is_draw_by_insufficient_material() or self.is_draw_by_repetition() or self.is_draw_by_fifty_move_rule()

    def get_winner(self):
        if self.is_checkmate():
            return -self.current_player  # The other player wins
        return 0  # Draw or game not over

    def is_checkmate(self):
        return self.is_in_check() and not self.get_legal_moves()

    def is_stalemate(self):
        return not self.is_in_check() and not self.get_legal_moves()

    def is_in_check(self):
        king_pos = np.where(self.board[:, :, 5] == self.current_player * 6)
        if len(king_pos[0]) == 0:
            return False  # No king found (shouldn't happen in a valid game)
        return self._is_square_attacked(king_pos[0][0], king_pos[1][0])

    def is_draw_by_insufficient_material(self):
        pieces = np.abs(self.board.sum(axis=2))
        if np.sum(pieces) == 2:  # Only kings left
            return True
        if np.sum(pieces) == 3 and (np.sum(np.abs(self.board[:, :, 2])) == 1 or np.sum(np.abs(self.board[:, :, 3])) == 1):
            return True  # King and bishop vs king or king and knight vs king
        return False

    def is_draw_by_repetition(self):
        # This would require keeping a history of positions, which we don't currently have
        # For simplicity, we'll return False, but in a full implementation, you'd check for threefold repetition
        return False

    def is_draw_by_fifty_move_rule(self):
        return self.halfmove_clock >= 100  # 50 full moves (100 half-moves)

    def get_state(self):
        return self.board.copy(), self.current_player
