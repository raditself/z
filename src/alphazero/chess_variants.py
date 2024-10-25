
import random

class StandardChess:
    def __init__(self):
        self.board_size = 8
        self.initial_state = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ]

def generate_chess960_position():
    # Place bishops
    light_square_bishop = random.randint(0, 3) * 2
    dark_square_bishop = random.randint(0, 3) * 2 + 1
    
    # Place queen
    remaining_squares = list(range(8))
    remaining_squares.remove(light_square_bishop)
    remaining_squares.remove(dark_square_bishop)
    queen = random.choice(remaining_squares)
    remaining_squares.remove(queen)
    
    # Place knights
    knight1 = random.choice(remaining_squares)
    remaining_squares.remove(knight1)
    knight2 = random.choice(remaining_squares)
    remaining_squares.remove(knight2)
    
    # Place rooks and king
    rook1, king, rook2 = remaining_squares
    
    # Create the position
    position = [0] * 8
    position[light_square_bishop] = 3  # Bishop
    position[dark_square_bishop] = 3  # Bishop
    position[queen] = 5  # Queen
    position[knight1] = 2  # Knight
    position[knight2] = 2  # Knight
    position[rook1] = 4  # Rook
    position[king] = 6  # King
    position[rook2] = 4  # Rook
    
    return position

def setup_chess960_board():
    board = [[0] * 8 for _ in range(8)]
    
    # Set up the back rank for white
    white_back_rank = generate_chess960_position()
    board[7] = white_back_rank
    
    # Set up pawns for white
    board[6] = [1] * 8
    
    # Set up the back rank for black (mirror of white)
    black_back_rank = [-piece for piece in white_back_rank]
    board[0] = black_back_rank
    
    # Set up pawns for black
    board[1] = [-1] * 8
    
    return board


class MiniChess:
    def __init__(self):
        self.board_size = 5
        self.initial_state = [
            ['r', 'n', 'b', 'q', 'k'],
            ['p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K']
        ]

class GrandChess:
    def __init__(self):
        self.board_size = 10
        self.initial_state = [
            ['r', 'n', 'b', 'q', 'k', 'c', 'a', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'C', 'A', 'B', 'N', 'R']
        ]
        # 'c' represents Chancellor (Rook + Knight)
        # 'a' represents Archbishop (Bishop + Knight)

class ThreeCheckChess:
    def __init__(self):
        self.board_size = 8
        self.initial_state = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ]
        self.check_count = {'white': 0, 'black': 0}

    def is_game_over(self):
        return self.check_count['white'] >= 3 or self.check_count['black'] >= 3

class KingOfTheHillChess:
    def __init__(self):
        self.board_size = 8
        self.initial_state = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ]
        self.center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]

    def is_game_over(self):
        for square in self.center_squares:
            if self.initial_state[square[0]][square[1]] in ['K', 'k']:
                return True
        return False

class Chess960:
    def __init__(self):
        self.board_size = 8
        self.initial_state = setup_chess960_board()

def get_chess_variant(variant_name):
    if variant_name == 'standard':
        return StandardChess()
    elif variant_name == 'mini':
        return MiniChess()
    elif variant_name == 'grand':
        return GrandChess()
    elif variant_name == 'three_check':
        return ThreeCheckChess()
    elif variant_name == 'king_of_the_hill':
        return KingOfTheHillChess()
    elif variant_name == 'chess960':
        return Chess960()
    else:
        raise ValueError("Unknown chess variant: " + str(variant_name))
