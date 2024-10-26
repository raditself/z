
import logging
import os
import sys
import time
from collections import deque
from pickle import Pickle, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

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

    def learn(self):
        for i in range(1, self.args.numIters + 1):
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

            # Evaluation
            self.mcts = MCTS(self.game, self.nnet, self.args)
            pnet = self.nnet.__class__(self.game)
            pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, pnet, self.args)
            
            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(self.mcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self):
        return 'checkpoint_' + str(self.iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickle.dump(self.trainExamplesHistory, f)

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

    def online_integration(self, args):
        log.info("Starting online integration...")
        online_integration = OnlineIntegration(self.game, args, self.nnet)

        if args.online_play:
            if args.tournament_id:
                log.info(f"Participating in Lichess tournament: {args.tournament_id}")
                online_integration.participate_in_tournament(args.tournament_id)
            else:
                log.info("Waiting for Lichess games...")
                while True:
                    # You might want to implement a method to check for incoming challenges
                    game_id = online_integration.lichess.check_for_challenges()
                    if game_id:
                        log.info(f"Starting game: {game_id}")
                        online_integration.play_lichess_game(game_id)
                    else:
                        time.sleep(10)  # Wait for 10 seconds before checking again

        if args.online_analysis:
            log.info("Starting real-time analysis of Lichess games...")
            while True:
                # You might want to implement a method to get ongoing games
                game_id = online_integration.lichess.get_ongoing_game()
                if game_id:
                    log.info(f"Analyzing game: {game_id}")
                    online_integration.real_time_analysis(game_id)
                else:
                    time.sleep(10)  # Wait for 10 seconds before checking again

        log.info("Online integration finished.")
