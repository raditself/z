
import chess
import chess.pgn
import io

class GameAnalysis:
    def __init__(self, ai):
        self.ai = ai

    def load_pgn(self, pgn_string):
        pgn = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn)
        return game

    def analyze_game(self, game):
        board = game.board()
        analysis = []

        for move in game.mainline_moves():
            board.push(move)
            evaluation = self.ai.evaluate_board(board)
            analysis.append({
                'move': move,
                'fen': board.fen(),
                'evaluation': evaluation
            })

        return analysis

    def get_critical_positions(self, analysis, threshold=200):
        critical_positions = []
        prev_eval = 0

        for pos in analysis:
            eval_diff = abs(pos['evaluation'] - prev_eval)
            if eval_diff > threshold:
                critical_positions.append(pos)
            prev_eval = pos['evaluation']

        return critical_positions

    def suggest_improvements(self, game, analysis):
        board = game.board()
        improvements = []

        for i, move_analysis in enumerate(analysis):
            if i % 2 == 0:  # Only analyze player's moves
                board.push(move_analysis['move'])
                best_move = self.ai.get_best_move(board)
                if best_move != move_analysis['move']:
                    improvements.append({
                        'move_number': i // 2 + 1,
                        'player_move': move_analysis['move'],
                        'suggested_move': best_move,
                        'evaluation_diff': self.ai.evaluate_board(board) - move_analysis['evaluation']
                    })
                board.pop()
            else:
                board.push(move_analysis['move'])

        return improvements
