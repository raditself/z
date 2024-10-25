
import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle, choice
import chess

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set up the distribution strategy
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy()
        print(f'Number of devices: {strategy.num_replicas_in_sync}')
    except RuntimeError as e:
        print(e)
else:
    strategy = tf.distribute.get_strategy()  # Default strategy
    print("No GPUs available. Using default strategy.")

from .game import Game
from .mcts import MCTS
from .model import get_model, get_data_loader
from .endgame_tablebase import EndgameTablebase
from .arena import Arena
from .utils import dotdict
from .neural_architecture_search import NeuralArchitectureSearch

log = logging.getLogger(__name__)

class Coach():
    def __init__(self, game: Game, args):
        self.game = game
        self.args = args
        with strategy.scope():
            self.nnet = get_model()
            self.pnet = get_model()  # the competitor network
        self.mcts = MCTS(self.game, self.args, self.nnet)
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False
        self.endgame_tablebase = EndgameTablebase()

        # Set up checkpointing
        self.checkpoint_dir = './checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(model=self.nnet, optimizer=tf.keras.optimizers.Adam())
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

        # Restore the latest checkpoint if it exists
        if self.manager.latest_checkpoint:
            self.checkpoint.restore(self.manager.latest_checkpoint)
            print(f"Restored from {self.manager.latest_checkpoint}")

        self.loss_history = []

    def executeEpisode(self):
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp, time_left=self.args.moveTime)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        for i in range(1, self.args.numIters + 1):
            log.info(f'Starting Iter #{i} ...')
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.args, self.nnet)   # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(i - 1)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

        # Save current model
        save_path = self.manager.save()
        print(f"Saved checkpoint for step {i}: {save_path}")
        # Load into previous network
        self.pnet.set_weights(self.nnet.get_weights())
        pmcts = MCTS(self.game, self.args, self.pnet)

        self.train(trainExamples)
        nmcts = MCTS(self.game, self.args, self.nnet)

        log.info('PITTING AGAINST PREVIOUS VERSION')
        arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0, time_left=self.args.moveTime)),
                      lambda x: np.argmax(nmcts.getActionProb(x, temp=0, time_left=self.args.moveTime)), self.game)
        pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

        log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
        if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
            log.info('REJECTING NEW MODEL')
            self.nnet.set_weights(self.pnet.get_weights())
        else:
            log.info('ACCEPTING NEW MODEL')
            save_path = self.manager.save()
            print(f"Saved checkpoint for step {i}: {save_path}")

        # Endgame tablebase training phase
        if i % self.args.endgameTrainingFrequency == 0:
            log.info('STARTING ENDGAME TABLEBASE TRAINING PHASE')
            self.train_from_tablebase()

        self.plot_loss()

    def train(self, examples):
        with strategy.scope():
            initial_learning_rate = 0.001
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=1000,
                decay_rate=0.96,
                staircase=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            
            @tf.function
            def train_step(boards, target_policies, target_values):
                with tf.GradientTape() as tape:
                    policy_output, value_output = self.nnet(boards, training=True)
                    policy_loss = loss_fn(target_policies, policy_output)
                    value_loss = tf.keras.losses.mean_squared_error(target_values, value_output)
                    total_loss = policy_loss + value_loss
                grads = tape.gradient(total_loss, self.nnet.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.nnet.trainable_variables))
                return total_loss

            for epoch in range(self.args.epochs):
                self.nnet.trainable = True
                dataset = get_data_loader(examples, batch_size=self.args.batch_size)
                total_loss = 0
                num_batches = 0
                for batch in dataset:
                    boards, target_policies, target_values = batch
                    per_replica_losses = strategy.run(train_step, args=(boards, target_policies, target_values))
                    total_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                    num_batches += 1
                avg_loss = total_loss / num_batches
                self.loss_history.append(avg_loss)
                current_lr = optimizer._decayed_lr(tf.float32).numpy()
                log.info(f'Epoch {epoch + 1}/{self.args.epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')

                # Save checkpoint after each epoch
                save_path = self.manager.save()
                print(f"Saved checkpoint for epoch {epoch+1}: {save_path}")

                # Evaluate model performance periodically
                if (epoch + 1) % 10 == 0:
                    self.evaluate_model()

    def evaluate_model(self):
        num_games = 20
        opponent = MCTS(self.game, self.args, self.pnet)  # Use previous network as opponent
        mcts = MCTS(self.game, self.args, self.nnet)
        arena = Arena(
            lambda x: np.argmax(mcts.getActionProb(x, temp=0)),
            lambda x: np.argmax(opponent.getActionProb(x, temp=0)),
            self.game
        )
        wins, losses, draws = arena.playGames(num_games)
        win_rate = (wins + 0.5 * draws) / num_games
        log.info(f'Evaluation: Win rate against previous version: {win_rate:.2f} (Wins: {wins}, Losses: {losses}, Draws: {draws})')
        return win_rate

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('training_loss.png')
        plt.close()

    def train_from_tablebase(self):
        endgame_examples = []
        for _ in range(self.args.numEndgameExamples):
            board = self.generate_random_endgame_position()
            wdl = self.endgame_tablebase.probe_wdl(board.fen())
            if wdl is not None:
                canonical_board = self.game.getCanonicalForm(board, 1)
                pi = self.generate_tablebase_policy(board)
                endgame_examples.append((canonical_board, pi, wdl / 2))  # Normalize value to [-1, 1]

        log.info(f'Generated {len(endgame_examples)} endgame examples')
        self.nnet.train(endgame_examples)

    def generate_random_endgame_position(self):
        pieces = ['K', 'k', 'Q', 'q', 'R', 'r', 'B', 'b', 'N', 'n', 'P', 'p']
        board = chess.Board.empty()

        # Always place kings
        board.set_piece_at(choice(range(64)), chess.Piece.from_symbol('K'))
        board.set_piece_at(choice([i for i in range(64) if board.piece_at(i) is None]), chess.Piece.from_symbol('k'))

        # Add 1-5 random pieces
        for _ in range(choice(range(1, 6))):
            square = choice([i for i in range(64) if board.piece_at(i) is None])
            piece = chess.Piece.from_symbol(choice(pieces[2:]))  # Exclude kings
            board.set_piece_at(square, piece)

        board.turn = choice([chess.WHITE, chess.BLACK])
        return board

    def generate_tablebase_policy(self, board):
        legal_moves = list(board.legal_moves)
        policy = [0] * self.game.getActionSize()
        
        best_move = self.endgame_tablebase.get_best_move(board)
        if best_move:
            best_move_index = legal_moves.index(best_move)
            policy[self.game.moveToAction(board, best_move)] = 1
        else:
            # If no best move found, use a uniform distribution over legal moves
            for move in legal_moves:
                policy[self.game.moveToAction(board, move)] = 1 / len(legal_moves)
        
        return policy

    # ... [rest of the Coach class remains unchanged]

if __name__ == "__main__":
    args = dotdict({
        'numIters': 1000,
        'numEps': 100,
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 25,
        'arenaCompare': 40,
        'cpuct': 1,

        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,

        'endgameTrainingFrequency': 10,  # Every 10 iterations
        'numEndgameExamples': 10000,  # Number of endgame positions to generate for training
        'moveTime': 30,  # Time allowed for each move in seconds
    })

    game = Game()
    nnet = NeuralArchitectureSearch(game)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(game, nnet, args)
    if args.load_model:
        log.info("Load trainExamples")
        c.loadTrainExamples()
    c.learn()
