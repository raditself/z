import random

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