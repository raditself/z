
import chess

class EndgamePatternRecognizer:
    def __init__(self):
        self.patterns = {
            'KQvK': self.kq_vs_k,
            'KRvK': self.kr_vs_k,
            'KBNvK': self.kbn_vs_k,
            'KPvK': self.kp_vs_k,
        }

    def recognize_pattern(self, board):
        for pattern_name, pattern_func in self.patterns.items():
            if pattern_func(board):
                return pattern_name
        return None

    def kq_vs_k(self, board):
        return (len(board.pieces(chess.QUEEN, chess.WHITE)) == 1 and
                len(board.pieces(chess.KING, chess.WHITE)) == 1 and
                len(board.pieces(chess.KING, chess.BLACK)) == 1 and
                len(board.piece_map()) == 3)

    def kr_vs_k(self, board):
        return (len(board.pieces(chess.ROOK, chess.WHITE)) == 1 and
                len(board.pieces(chess.KING, chess.WHITE)) == 1 and
                len(board.pieces(chess.KING, chess.BLACK)) == 1 and
                len(board.piece_map()) == 3)

    def kbn_vs_k(self, board):
        return (len(board.pieces(chess.BISHOP, chess.WHITE)) == 1 and
                len(board.pieces(chess.KNIGHT, chess.WHITE)) == 1 and
                len(board.pieces(chess.KING, chess.WHITE)) == 1 and
                len(board.pieces(chess.KING, chess.BLACK)) == 1 and
                len(board.piece_map()) == 4)

    def kp_vs_k(self, board):
        return (len(board.pieces(chess.PAWN, chess.WHITE)) == 1 and
                len(board.pieces(chess.KING, chess.WHITE)) == 1 and
                len(board.pieces(chess.KING, chess.BLACK)) == 1 and
                len(board.piece_map()) == 3)

def apply_endgame_strategy(board, pattern):
    if pattern == 'KQvK':
        return kq_vs_k_strategy(board)
    elif pattern == 'KRvK':
        return kr_vs_k_strategy(board)
    elif pattern == 'KBNvK':
        return kbn_vs_k_strategy(board)
    elif pattern == 'KPvK':
        return kp_vs_k_strategy(board)
    else:
        return None

def kq_vs_k_strategy(board):
    # Implementation of KQ vs K endgame strategy
    king = board.king(chess.WHITE)
    enemy_king = board.king(chess.BLACK)
    queen = board.pieces(chess.QUEEN, chess.WHITE).pop()
    
    # Step 1: Restrict enemy king's movement
    if chess.square_distance(king, enemy_king) > 2:
        return chess.Move(queen, chess.square(chess.file_index(enemy_king), chess.rank_index(enemy_king) + 1))
    
    # Step 2: Bring our king closer
    return chess.Move(king, chess.square(chess.file_index(enemy_king), chess.rank_index(enemy_king) + 2))

def kr_vs_k_strategy(board):
    # Implementation of KR vs K endgame strategy
    king = board.king(chess.WHITE)
    enemy_king = board.king(chess.BLACK)
    rook = board.pieces(chess.ROOK, chess.WHITE).pop()
    
    # Step 1: Use rook to restrict enemy king's movement
    if chess.square_distance(rook, enemy_king) > 2:
        return chess.Move(rook, chess.square(chess.file_index(enemy_king), chess.rank_index(rook)))
    
    # Step 2: Bring our king closer
    return chess.Move(king, chess.square(chess.file_index(enemy_king), chess.rank_index(enemy_king) + 2))

def kbn_vs_k_strategy(board):
    # Implementation of KBN vs K endgame strategy
    king = board.king(chess.WHITE)
    enemy_king = board.king(chess.BLACK)
    bishop = board.pieces(chess.BISHOP, chess.WHITE).pop()
    knight = board.pieces(chess.KNIGHT, chess.WHITE).pop()
    
    # Step 1: Drive enemy king to a corner
    if not chess.square_mirror(enemy_king) in [chess.A8, chess.H8]:
        return chess.Move(king, chess.square(chess.file_index(enemy_king), chess.rank_index(enemy_king) + 1))
    
    # Step 2: Position bishop and knight for checkmate
    if chess.square_distance(bishop, enemy_king) > 2:
        return chess.Move(bishop, chess.square(chess.file_index(enemy_king) - 1, chess.rank_index(enemy_king) - 1))
    return chess.Move(knight, chess.square(chess.file_index(enemy_king) - 2, chess.rank_index(enemy_king) - 1))

def kp_vs_k_strategy(board):
    # Implementation of KP vs K endgame strategy
    king = board.king(chess.WHITE)
    enemy_king = board.king(chess.BLACK)
    pawn = board.pieces(chess.PAWN, chess.WHITE).pop()
    
    # Step 1: Advance pawn if it's safe
    pawn_advance = chess.Move(pawn, chess.square(chess.file_index(pawn), chess.rank_index(pawn) - 1))
    if pawn_advance in board.legal_moves and chess.square_distance(enemy_king, pawn) > 1:
        return pawn_advance
    
    # Step 2: Use king to support pawn advancement
    return chess.Move(king, chess.square(chess.file_index(pawn), chess.rank_index(pawn) + 1))
