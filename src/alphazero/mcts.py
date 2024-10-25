
import math
import numpy as np
import multiprocessing
from multiprocessing import Pool
from .game import Game
from .endgame_tablebase import EndgameTablebase

class MCTS:
    def __init__(self, game: Game, args, model):
        self.game = game
        self.args = args
        self.model = model
        self.Qsa = {}  # stores Q values for s,a
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}   # stores #times board s was visited
        self.Ps = {}   # stores initial policy (returned by neural net)
        self.Es = {}   # stores game.getGameEnded ended for board s
        self.Vs = {}   # stores game.getValidMoves for board s
        self.tablebase = EndgameTablebase()
        self.pool = Pool(processes=multiprocessing.cpu_count())

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def parallel_search(self, canonicalBoard):
        results = self.pool.map(self.search, [canonicalBoard] * self.args.numMCTSSims)
        for v in results:
            s = self.game.stringRepresentation(canonicalBoard)
            self.Ns[s] += 1

    def getActionProb(self, canonicalBoard, temp=1, time_left=None):
        if self.tablebase.should_use_tablebase(canonicalBoard, time_left):
            best_move = self.tablebase.get_best_move(canonicalBoard)
            if best_move:
                probs = [0] * self.game.getActionSize()
                probs[self.game.moveToAction(canonicalBoard, best_move)] = 1
                return probs

        self.parallel_search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        if self.tablebase.should_use_tablebase(canonicalBoard, time_left=None):
            wdl = self.tablebase.probe_wdl(canonicalBoard.fen())
            if wdl is not None:
                return -wdl / 2  # Convert to our value range (-1 to 1)

        if s not in self.Ps:
            self.Ps[s], v = self.model.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + self.args.EPS)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
