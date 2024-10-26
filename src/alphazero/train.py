
import argparse
from .game import Game
from .nnet import NNetWrapper
from .coach import Coach
from .utils import *
from .online_integration import OnlineIntegration
from .data_handler import DataHandler

def main():
    parser = argparse.ArgumentParser(description='Train and play with AlphaZero')
    parser.add_argument('--numIters', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--numEps', type=int, default=100, help='Number of episodes per iteration')
    parser.add_argument('--tempThreshold', type=int, default=15, help='Temperature threshold')
    parser.add_argument('--updateThreshold', type=float, default=0.6, help='Update threshold')
    parser.add_argument('--maxlenOfQueue', type=int, default=200000, help='Max length of queue')
    parser.add_argument('--numMCTSSims', type=int, default=25, help='Number of MCTS simulations')
    parser.add_argument('--arenaCompare', type=int, default=40, help='Number of games to play in arena')
    parser.add_argument('--cpuct', type=float, default=1.0, help='CPUCT parameter')
    parser.add_argument('--checkpoint', type=str, default='./temp/', help='Checkpoint directory')
    parser.add_argument('--load_model', type=bool, default=False, help='Load trained model')
    parser.add_argument('--load_folder_file', type=tuple, default=('/dev/models/8x100x50','best.pth.tar'), help='Folder and file to load model from')
    parser.add_argument('--numItersForTrainExamplesHistory', type=int, default=20, help='Number of iterations for train examples history')
    parser.add_argument('--moveTime', type=int, default=1, help='Time limit for each move (in seconds)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for storing training data')

    # New arguments for neural architecture search
    parser.add_argument('--perform_nas', type=bool, default=False, help='Perform Neural Architecture Search')
    parser.add_argument('--nas_epochs', type=int, default=5, help='Number of epochs for NAS')

    # New arguments for online integration
    parser.add_argument('--lichess_token', type=str, default='', help='Lichess API token')
    parser.add_argument('--chess_com_username', type=str, default='', help='Chess.com username')
    parser.add_argument('--chess_com_password', type=str, default='', help='Chess.com password')
    parser.add_argument('--online_play', type=bool, default=False, help='Enable online play')
    parser.add_argument('--online_analysis', type=bool, default=False, help='Enable online analysis')
    parser.add_argument('--tournament_id', type=str, default='', help='Lichess tournament ID to participate in')

    args = parser.parse_args()

    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', NNetWrapper.__name__)
    nnet = NNetWrapper(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Initializing DataHandler...')
    data_handler = DataHandler(args.data_dir)

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args, data_handler)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    if args.online_play or args.online_analysis:
        c.online_integration(args)
    else:
        log.info('Starting the learning process 🎉')
        c.learn()

if __name__ == "__main__":
    main()
