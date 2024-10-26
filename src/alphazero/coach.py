

import logging
import os
import sys
import time
from collections import deque
from pickle import Pickle, Unpickler
from random import shuffle
import numpy as np
from tqdm import tqdm
import torch

from .game import Game
from .mcts import MCTS
from .arena import Arena
from .online_integration import OnlineIntegration

log = logging.getLogger(__name__)

class Coach():
    def __init__(self, game: Game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.current_iteration = 0

    def executeEpisode(self):
        trainExamples = []
        currentPlayer = 1
        episodeStep = 0

        state = self.game.getInitBoard()
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(state, currentPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, currentPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            state, currentPlayer = self.game.getNextState(state, currentPlayer, action)

            r = self.game.getGameEnded(state, currentPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != currentPlayer))) for x in trainExamples]

    def save_checkpoint(self, filename):
        checkpoint = {
            'iteration': self.current_iteration,
            'model_state_dict': self.nnet.model.state_dict(),
            'optimizer_state_dict': self.nnet.optimizer.state_dict(),
            'trainExamplesHistory': self.trainExamplesHistory,
            'args': self.args
        }
        torch.save(checkpoint, filename)
        log.info(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint file {filename} not found")
        
        checkpoint = torch.load(filename)
        self.current_iteration = checkpoint['iteration']
        self.nnet.model.load_state_dict(checkpoint['model_state_dict'])
        self.nnet.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.trainExamplesHistory = checkpoint['trainExamplesHistory']
        self.args = checkpoint['args']
        log.info(f"Checkpoint loaded from {filename}")

    def learn(self):
        for i in range(self.current_iteration + 1, self.args.numIters + 1):
            self.current_iteration = i
            log.info(f'Starting Iteration {i}')
            
            # Self-play
            self.mcts = MCTS(self.game, self.nnet, self.args)
            trainExamples = []
            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                trainExamples.extend(self.executeEpisode())

            # Save the iteration examples to the history 
            self.trainExamplesHistory.append(trainExamples)
            
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            
            # Shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # Training
            self.nnet.train_network(trainExamples, batch_size=self.args.batch_size, epochs=self.args.epochs)

            # Save checkpoint
            checkpoint_filename = os.path.join(self.args.checkpoint, f'checkpoint_iter_{i}.pth.tar')
            self.save_checkpoint(checkpoint_filename)

            # Evaluate the model (you can add your evaluation logic here)
            # ...

        log.info('Training completed.')

