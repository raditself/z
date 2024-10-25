
import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle, choice
import chess

import numpy as np
import torch
from tqdm import tqdm

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
        self.nnet = get_model()
        self.pnet = get_model()  # the competitor network
        self.mcts = MCTS(self.game, self.args, self.nnet)
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False
        self.endgame_tablebase = EndgameTablebase()

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
            torch.save(self.nnet.state_dict(), os.path.join(self.args.checkpoint, 'temp.pth.tar'))
            # Load into previous network
            self.pnet.load_state_dict(torch.load(os.path.join(self.args.checkpoint, 'temp.pth.tar')))
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
                self.nnet.load_state_dict(torch.load(os.path.join(self.args.checkpoint, 'temp.pth.tar')))
            else:
                log.info('ACCEPTING NEW MODEL')
                torch.save(self.nnet.state_dict(), os.path.join(self.args.checkpoint, self.getCheckpointFile(i)))
                torch.save(self.nnet.state_dict(), os.path.join(self.args.checkpoint, 'best.pth.tar'))

            # Endgame tablebase training phase
            if i % self.args.endgameTrainingFrequency == 0:
                log.info('STARTING ENDGAME TABLEBASE TRAINING PHASE')
                self.train_from_tablebase()

    def train(self, examples):
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=0.001)
        for epoch in range(self.args.epochs):
            self.nnet.train()
            data_loader = get_data_loader(examples, batch_size=self.args.batch_size)
            for batch_idx, (boards, target_policies, target_values) in enumerate(data_loader):
                boards, target_policies, target_values = boards.to(self.args.device), target_policies.to(self.args.device), target_values.to(self.args.device)

                optimizer.zero_grad()
                policy_output, value_output = self.nnet(boards)
                policy_loss = torch.nn.functional.cross_entropy(policy_output, target_policies)
                value_loss = torch.nn.functional.mse_loss(value_output.squeeze(), target_values)
                total_loss = policy_loss + value_loss
                total_loss.backward()
                optimizer.step()

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
