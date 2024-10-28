
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
from .curriculum_learning import CurriculumLearning

log = logging.getLogger(__name__)

class Coach():
    def __init__(self, game: Game, nnet, args, curriculum: CurriculumLearning):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.current_iteration = 0
        self.curriculum = curriculum
        self.style_history = []  # Store playing style data

    def collect_style_metrics(self, trainExamples):
        # Implement style metrics collection here
        # This is a placeholder implementation and should be adapted based on your specific game and requirements
        num_moves = len(trainExamples)
        avg_move_value = sum(example[2] for example in trainExamples) / num_moves
        aggressive_moves = sum(1 for example in trainExamples if example[2] > 0.7)  # Assuming moves with high probability are aggressive
        defensive_moves = sum(1 for example in trainExamples if example[2] < 0.3)  # Assuming moves with low probability are defensive
        
        return {
            'num_moves': num_moves,
            'avg_move_value': avg_move_value,
            'aggressive_moves': aggressive_moves,
            'defensive_moves': defensive_moves,
        }

    def executeEpisode(self):
        trainExamples = []
        currentPlayer = 1
        episodeStep = 0

        state = self.game.getInitBoard()
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(state, currentPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            # Apply curriculum learning
            canonicalBoard = self.curriculum.adjust_game_complexity(canonicalBoard)

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
            'args': self.args,
            'curriculum_state': self.curriculum.get_state(),
            'style_history': self.style_history
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
        self.curriculum.set_state(checkpoint['curriculum_state'])
        self.style_history = checkpoint.get('style_history', [])
        log.info(f"Checkpoint loaded from {filename}")

    def learn(self):
        for i in range(self.current_iteration + 1, self.args.numIters + 1):
            self.current_iteration = i
            log.info(f'Starting Iteration {i}, Curriculum Complexity: {self.curriculum.get_current_complexity():.2f}')
            
            # Self-play
            self.mcts = MCTS(self.game, self.nnet, self.args)
            trainExamples = []
            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                trainExamples.extend(self.executeEpisode())

            # Collect and store style metrics
            style_metrics = self.collect_style_metrics(trainExamples)
            self.style_history.append(style_metrics)

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

            # Training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)
            
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            # Step the curriculum after each iteration
            self.curriculum.step()

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickle.dump(self.trainExamplesHistory, f)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
