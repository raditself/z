
import chess
import chess.syzygy
from functools import lru_cache
import os
import mmap
import numpy as np
import zlib
from collections import Counter
import time

class CompressedTablebase:
    def __init__(self, tablebase_path):
        self.tablebase_path = tablebase_path
        self.compressed_data = {}
        self.load_and_compress_tablebases()

    def load_and_compress_tablebases(self):
        for root, dirs, files in os.walk(self.tablebase_path):
            for file in files:
                if file.endswith('.rtbw') or file.endswith('.rtbz'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        data = f.read()
                        compressed_data = zlib.compress(data)
                        self.compressed_data[file] = compressed_data

    def decompress_file(self, file_name):
        return zlib.decompress(self.compressed_data[file_name])

class EndgameTablebase:
    def __init__(self):
        self.tablebase_path = "/home/user/z/data/syzygy"
        self.compressed_tablebase = CompressedTablebase(self.tablebase_path)
        self.tablebase = chess.syzygy.Tablebase()
        self.memory_mapped_files = {}
        self.loaded_piece_count = 0
        self.load_tablebases(3)  # Initially load 3-piece tablebases

    def load_tablebases(self, piece_count):
        if piece_count <= self.loaded_piece_count:
            return

        for root, dirs, files in os.walk(self.tablebase_path):
            for file in files:
                if file.endswith('.rtbw') or file.endswith('.rtbz'):
                    file_pieces = int(file.split('-')[0])
                    if self.loaded_piece_count < file_pieces <= piece_count:
                        self.load_tablebase_file(os.path.join(root, file))

        self.loaded_piece_count = piece_count

    def load_tablebase_file(self, file_path):
        file_name = os.path.basename(file_path)
        decompressed_data = self.compressed_tablebase.decompress_file(file_name)
        mmap_file = mmap.mmap(-1, len(decompressed_data))
        mmap_file.write(decompressed_data)
        mmap_file.seek(0)
        self.memory_mapped_files[file_name] = mmap_file
        self.tablebase.add_directory(mmap_file)

    @lru_cache(maxsize=100000)
    def probe_wdl(self, board_fen):
        board = chess.Board(board_fen)
        try:
            return self.tablebase.probe_wdl(board)
        except (chess.syzygy.MissingTableError, ValueError):
            return None

    @lru_cache(maxsize=100000)
    def probe_dtz(self, board_fen):
        board = chess.Board(board_fen)
        try:
            return self.tablebase.probe_dtz(board)
        except (chess.syzygy.MissingTableError, ValueError):
            return None

    def batch_probe_wdl(self, board_fens):
        results = []
        for fen in board_fens:
            results.append(self.probe_wdl(fen))
        return results

    def recognize_endgame_pattern(self, board):
        piece_counts = Counter(board.piece_map().values())
        total_pieces = sum(piece_counts.values())
        
        patterns = {
            "KPK": (total_pieces == 3 and piece_counts[chess.PAWN] == 1),
            "KBNK": (total_pieces == 4 and piece_counts[chess.BISHOP] == 1 and piece_counts[chess.KNIGHT] == 1),
            "KBBK": (total_pieces == 4 and piece_counts[chess.BISHOP] == 2),
            "KRKP": (total_pieces == 4 and piece_counts[chess.ROOK] == 1 and piece_counts[chess.PAWN] == 1),
            "KQKP": (total_pieces == 4 and piece_counts[chess.QUEEN] == 1 and piece_counts[chess.PAWN] == 1),
        }
        
        for pattern, condition in patterns.items():
            if condition:
                return pattern
        
        return "Unknown"

    def apply_endgame_principles(self, board, pattern):
        if pattern == "KPK":
            return self.kpk_principle(board)
        elif pattern == "KBNK":
            return self.kbnk_principle(board)
        elif pattern == "KBBK":
            return self.kbbk_principle(board)
        elif pattern == "KRKP":
            return self.krkp_principle(board)
        elif pattern == "KQKP":
            return self.kqkp_principle(board)
        else:
            return None

    def kpk_principle(self, board):
        # Implement KPK endgame principle
        pass

    def kbnk_principle(self, board):
        # Implement KBNK endgame principle
        pass

    def kbbk_principle(self, board):
        # Implement KBBK endgame principle
        pass

    def krkp_principle(self, board):
        # Implement KRKP endgame principle
        pass

    def kqkp_principle(self, board):
        # Implement KQKP endgame principle
        pass

    def get_best_move(self, board):
        if not board.legal_moves:
            return None  # No legal moves available (game has ended)

        piece_count = sum(1 for _ in board.piece_map().values())
        
        if piece_count <= 7:
            self.load_tablebases(piece_count)  # Ensure appropriate tablebases are loaded

            pattern = self.recognize_endgame_pattern(board)
            principle_move = self.apply_endgame_principles(board, pattern)
            
            if principle_move:
                return principle_move

            # If no principle applies, use the tablebase probing
            legal_moves = list(board.legal_moves)
            board_fens = []
            for move in legal_moves:
                board.push(move)
                board_fens.append(board.fen())
                board.pop()

            wdl_values = self.batch_probe_wdl(board_fens)
            
            best_move = None
            best_wdl = -3
            for move, wdl in zip(legal_moves, wdl_values):
                if wdl is not None:
                    wdl = -wdl
                    if wdl > best_wdl:
                        best_wdl = wdl
                        best_move = move

            return best_move if best_move else np.random.choice(legal_moves)
        else:
            # For positions with more than 7 pieces, return a random legal move
            return np.random.choice(list(board.legal_moves))

    def __del__(self):
        for mmap_file in self.memory_mapped_files.values():
            mmap_file.close()

def benchmark_tablebase_loading():
    start_time = time.time()
    tablebase = EndgameTablebase()
    end_time = time.time()
    print(f"Time to load initial tablebases: {end_time - start_time:.2f} seconds")

    start_time = time.time()
    tablebase.load_tablebases(7)
    end_time = time.time()
    print(f"Time to load up to 7-piece tablebases: {end_time - start_time:.2f} seconds")

def benchmark_move_generation(tablebase, num_positions=1000):
    total_time = 0
    for _ in range(num_positions):
        board = chess.Board()
        for _ in range(np.random.randint(20, 60)):  # Random number of moves
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(np.random.choice(moves))

        start_time = time.time()
        move = tablebase.get_best_move(board)
        end_time = time.time()
        total_time += end_time - start_time

        if move is None and board.legal_moves:
            print(f"Warning: No move returned for a non-terminal position: {board.fen()}")

    avg_time = total_time / num_positions
    print(f"Average time to generate a move: {avg_time:.4f} seconds")

def run_benchmarks():
    print("Running benchmarks...")
    benchmark_tablebase_loading()
    tablebase = EndgameTablebase()
    benchmark_move_generation(tablebase)

if __name__ == "__main__":
    run_benchmarks()
