





import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import multiprocessing as mp
import pygame
import unittest
from abc import ABC, abstractmethod
from .alphazero_net import AlphaZeroNet
from .hierarchical_mcts import HierarchicalMCTS
from .meta_learning import MetaLearner
from .explainable_ai import ExplainableAI
from .transfer_learning import TransferLearning
from .style_analyzer import StyleAnalyzer
from .external_knowledge_integration import ExternalKnowledgeIntegrator, OpeningBook, EndgameTablebase

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Game(ABC):
    @abstractmethod
    def __init__(self, board_size):
        self.board_size = board_size
        self.action_size = board_size ** 2

    @abstractmethod
    def get_initial_state(self):
        pass

    @abstractmethod
    def get_next_state(self, state, action):
        pass

    @abstractmethod
    def get_valid_moves(self, state):
        pass

    @abstractmethod
    def is_game_over(self, state):
        pass

    @abstractmethod
    def get_game_result(self, state):
        pass

    @abstractmethod
    def get_encoded_state(self, state):
        pass

    @abstractmethod
    def string_representation(self, state):
        pass

class TicTacToe(Game):
    def __init__(self, board_size=3):
        super().__init__(board_size)

    def get_initial_state(self):
        return np.zeros((self.board_size, self.board_size), dtype=np.int8)

    def get_next_state(self, state, action):
        player = 1 if np.sum(state) % 2 == 0 else -1
        state = state.copy()
        state[action // self.board_size, action % self.board_size] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def is_game_over(self, state):
        for player in [1, -1]:
            if np.any(np.sum(state * player, axis=0) == self.board_size) or                np.any(np.sum(state * player, axis=1) == self.board_size) or                np.sum(np.diag(state) * player) == self.board_size or                np.sum(np.diag(np.fliplr(state)) * player) == self.board_size:
                return True
        return np.sum(self.get_valid_moves(state)) == 0

    def get_game_result(self, state):
        for player in [1, -1]:
            if np.any(np.sum(state * player, axis=0) == self.board_size) or                np.any(np.sum(state * player, axis=1) == self.board_size) or                np.sum(np.diag(state) * player) == self.board_size or                np.sum(np.diag(np.fliplr(state)) * player) == self.board_size:
                return player
        return 0

    def get_encoded_state(self, state):
        encoded = np.stack(
            (state == 1, state == -1, np.ones_like(state) * (np.sum(state) % 2 == 0))
        ).astype(np.float32)
        return torch.tensor(encoded)

    def string_representation(self, state):
        return state.tobytes()

class AlphaZero:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AlphaZeroNet(game.board_size, game.action_size).to(self.device)
        
        # Initialize external knowledge bases
        self.opening_book = OpeningBook(args['opening_book_path'])
        self.endgame_tablebase = EndgameTablebase(args['endgame_tablebase_path'])
        self.external_knowledge_integrator = ExternalKnowledgeIntegrator([self.opening_book, self.endgame_tablebase])
        
        self.mcts = HierarchicalMCTS(game, self.net, args, self.external_knowledge_integrator)
        self.meta_learner = MetaLearner(self.net, args)
        self.explainable_ai = ExplainableAI(self.net, game)
        self.transfer_learning = TransferLearning(self.net)
        self.style_analyzer = StyleAnalyzer()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args['lr'])

    def self_play(self, num_games):
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(self._play_single_game, range(num_games))
        
        examples = []
        for game_examples, trajectory in results:
            examples.extend(game_examples)
            self.style_analyzer.update_from_game(trajectory, game_examples[-1][2])
        
        return examples

    def _play_single_game(self, _):
        state = self.game.get_initial_state()
        trajectory = []
        examples = []
        
        while not self.game.is_game_over(state):
            action_probs = self.mcts.search(state)
            trajectory.append((state, action_probs))
            action = np.random.choice(self.game.action_size, p=action_probs)
            state = self.game.get_next_state(state, action)
        
        value = self.game.get_game_result(state)
        examples = [(s, p, value) for s, p in trajectory]
        
        return examples, trajectory

    def train(self, examples):
        self.net.train()
        for _ in range(self.args['epochs']):
            loss = self.meta_learner.train_step(examples)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def play(self, state):
        action_probs = self.mcts.search(state)
        return np.argmax(action_probs)

    def explain_decision(self, state):
        return self.explainable_ai.explain(state)

    def adapt_to_new_game(self, new_game):
        self.transfer_learning.adapt(new_game)

    def generate_style_report(self):
        return self.style_analyzer.generate_report()

    def integrate_external_knowledge(self, state):
        return self.external_knowledge_integrator.integrate_knowledge(state)

    def save_model(self, filename):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        logger.info(f"Model saved to {filename}")

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {filename}")

    def evaluate(self, num_games=100, opponent='random'):
        wins = 0
        for _ in range(num_games):
            state = self.game.get_initial_state()
            while not self.game.is_game_over(state):
                if np.sum(state) % 2 == 0:  # AlphaZero's turn
                    action = self.play(state)
                else:  # Opponent's turn
                    if opponent == 'random':
                        valid_moves = self.game.get_valid_moves(state)
                        action = np.random.choice(np.where(valid_moves)[0])
                    elif opponent == 'minimax':
                        action = self.minimax(state, depth=3, maximizing=False)[1]
                state = self.game.get_next_state(state, action)
            if self.game.get_game_result(state) == 1:
                wins += 1
        return wins / num_games

    def minimax(self, state, depth, maximizing):
        if depth == 0 or self.game.is_game_over(state):
            return self.game.get_game_result(state), None

        valid_moves = self.game.get_valid_moves(state)
        if maximizing:
            best_value = float('-inf')
            best_move = None
            for action in np.where(valid_moves)[0]:
                next_state = self.game.get_next_state(state, action)
                value, _ = self.minimax(next_state, depth - 1, False)
                if value > best_value:
                    best_value = value
                    best_move = action
            return best_value, best_move
        else:
            best_value = float('inf')
            best_move = None
            for action in np.where(valid_moves)[0]:
                next_state = self.game.get_next_state(state, action)
                value, _ = self.minimax(next_state, depth - 1, True)
                if value < best_value:
                    best_value = value
                    best_move = action
            return best_value, best_move

class GUI:
    def __init__(self, game, alphazero):
        self.game = game
        self.alphazero = alphazero
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("TicTacToe vs AlphaZero")
        self.font = pygame.font.Font(None, 36)

    def draw_board(self, state):
        self.screen.fill((255, 255, 255))
        for i in range(1, 3):
            pygame.draw.line(self.screen, (0, 0, 0), (i * 133, 0), (i * 133, 400), 2)
            pygame.draw.line(self.screen, (0, 0, 0), (0, i * 133), (400, i * 133), 2)
        for i in range(3):
            for j in range(3):
                if state[i, j] == 1:
                    pygame.draw.line(self.screen, (255, 0, 0), (j * 133 + 20, i * 133 + 20), ((j + 1) * 133 - 20, (i + 1) * 133 - 20), 2)
                    pygame.draw.line(self.screen, (255, 0, 0), ((j + 1) * 133 - 20, i * 133 + 20), (j * 133 + 20, (i + 1) * 133 - 20), 2)
                elif state[i, j] == -1:
                    pygame.draw.circle(self.screen, (0, 0, 255), (j * 133 + 66, i * 133 + 66), 46, 2)

    def play_game(self):
        state = self.game.get_initial_state()
        player_turn = True
        while not self.game.is_game_over(state):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and player_turn:
                    x, y = event.pos
                    action = (y // 133) * 3 + (x // 133)
                    if self.game.get_valid_moves(state)[action]:
                        state = self.game.get_next_state(state, action)
                        player_turn = False
            
            if not player_turn and not self.game.is_game_over(state):
                action = self.alphazero.play(state)
                state = self.game.get_next_state(state, action)
                player_turn = True
            
            self.draw_board(state)
            pygame.display.flip()
        
        result = self.game.get_game_result(state)
        if result == 1:
            text = self.font.render("AlphaZero wins!", True, (0, 255, 0))
        elif result == -1:
            text = self.font.render("You win!", True, (0, 255, 0))
        else:
            text = self.font.render("It's a draw!", True, (0, 255, 0))
        self.screen.blit(text, (150, 180))
        pygame.display.flip()
        pygame.time.wait(3000)
        pygame.quit()

def get_args():
    args = {
        'num_iterations': 1000,
        'num_episodes': 100,
        'num_mcts_sims': 25,
        'cpuct': 1.0,
        'lr': 0.001,
        'epochs': 10,
        'batch_size': 64,
        'num_channels': 256,
        'opening_book_path': 'data/opening_book.json',
        'endgame_tablebase_path': 'data/endgame_tablebase.db',
        'external_knowledge_weight': 0.5,
        'epsilon': 1e-8,
    }
    return args

class TestAlphaZero(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToe()
        self.args = get_args()
        self.alphazero = AlphaZero(self.game, self.args)

    def test_initial_state(self):
        state = self.game.get_initial_state()
        self.assertTrue(np.all(state == 0))

    def test_game_over(self):
        state = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        self.assertTrue(self.game.is_game_over(state))

    def test_valid_moves(self):
        state = np.array([[1, 0, 1], [0, -1, 0], [0, 0, 0]])
        valid_moves = self.game.get_valid_moves(state)
        self.assertEqual(np.sum(valid_moves), 6)

    def test_self_play(self):
        examples = self.alphazero.self_play(num_games=10)
        self.assertGreater(len(examples), 0)

    def test_train(self):
        examples = self.alphazero.self_play(num_games=10)
        loss = self.alphazero.train(examples)
        self.assertIsInstance(loss, float)

    def test_save_load_model(self):
        self.alphazero.save_model("test_model.pth")
        self.assertTrue(os.path.exists("test_model.pth"))
        self.alphazero.load_model("test_model.pth")
        os.remove("test_model.pth")

if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False)

    # Main training loop
    game = TicTacToe()
    args = get_args()
    alphazero = AlphaZero(game, args)
    
    for i in range(args['num_iterations']):
        logger.info(f"Starting iteration {i+1}")
        examples = alphazero.self_play(args['num_episodes'])
        loss = alphazero.train(examples)
        logger.info(f"Iteration {i+1} completed. Loss: {loss:.4f}")
        
        if (i + 1) % 10 == 0:
            alphazero.save_model(f"models/alphazero_iteration_{i+1}.pth")
            win_rate_random = alphazero.evaluate(opponent='random')
            win_rate_minimax = alphazero.evaluate(opponent='minimax')
            logger.info(f"Evaluation after iteration {i+1}:")
            logger.info(f"Win rate against random opponent: {win_rate_random:.2f}")
            logger.info(f"Win rate against minimax opponent: {win_rate_minimax:.2f}")
        
        if (i + 1) % 100 == 0:
            style_report = alphazero.generate_style_report()
            logger.info(f"Style report after iteration {i+1}:\n{style_report}")

    # Final evaluation
    alphazero.save_model("models/alphazero_final.pth")
    final_win_rate_random = alphazero.evaluate(num_games=1000, opponent='random')
    final_win_rate_minimax = alphazero.evaluate(num_games=1000, opponent='minimax')
    logger.info("Final evaluation:")
    logger.info(f"Win rate against random opponent: {final_win_rate_random:.2f}")
    logger.info(f"Win rate against minimax opponent: {final_win_rate_minimax:.2f}")

    # Generate final style report
    final_style_report = alphazero.generate_style_report()
    logger.info(f"Final style report:\n{final_style_report}")

    logger.info("Training completed.")

    # Start GUI for human vs AI play
    gui = GUI(game, alphazero)
    gui.play_game()





