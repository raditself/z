import torch
from torch.utils.data import Dataset
import chess
import sqlite3
import numpy as np

class ChessDataset(Dataset):
    def __init__(self, db_path, train=True):
        self.db_path = db_path
        self.train = train
        self.conn = None
        self.cursor = None
        self.connect()
        
        # Get the total number of samples
        self.cursor.execute("SELECT COUNT(*) FROM evaluations")
        self.total_samples = self.cursor.fetchone()[0]
        
        # Use 80% for training, 20% for validation
        self.split_index = int(0.8 * self.total_samples)
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def __len__(self):
        return self.split_index if self.train else (self.total_samples - self.split_index)
    
    def __getitem__(self, idx):
        if not self.conn:
            self.connect()
        
        if not self.train:
            idx += self.split_index
        
        try:
            self.cursor.execute("SELECT fen, eval FROM evaluations LIMIT 1 OFFSET ?", (int(idx),))
            fen, eval_score = self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return None
        
        board = chess.Board(fen)
        state = self.board_to_tensor(board)
        eval_score = torch.tensor([float(eval_score)], dtype=torch.float32)
        
        return state, eval_score
    
    def board_to_tensor(self, board):
        pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
        state = torch.zeros(12, 8, 8, dtype=torch.float32)
        
        for i, piece in enumerate(pieces):
            for square in board.pieces(chess.PIECE_SYMBOLS.index(piece.lower()), piece.isupper()):
                rank, file = divmod(square, 8)
                state[i, rank, file] = 1
        
        return state.view(-1)  # Flatten the tensor

    def __del__(self):
        if self.conn:
            self.conn.close()

