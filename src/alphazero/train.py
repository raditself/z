
# Updated content for train.py
import argparse
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from game import Game
from nnet import NNetWrapper
from coach import Coach
from utils import *
from online_integration import OnlineIntegration
from data_handler import DataHandler
from model_evaluator import ModelEvaluator
from logging_system import setup_logger, log_decorator
from analysis_toolkit import AnalysisToolkit
from alphazero_net import AlphaZeroNet, ResNetAlphaZero, TransformerAlphaZero
from architecture_comparison import compare_architectures
from visualization import ArchitectureVisualizer
from curriculum_learning import CurriculumLearning
from curriculum_visualizer import CurriculumVisualizer
from game_phase_handler import GamePhaseHandler

def main():
    parser = argparse.ArgumentParser(description='Train and play with AlphaZero')
    # ... (keep all existing argument definitions)
    parser.add_argument('--architecture', type=str, choices=['original', 'resnet', 'transformer'], default='original', help='Neural network architecture to use')
    parser.add_argument('--compare_every', type=int, default=10, help='Compare architectures every N iterations')
    parser.add_argument('--comparison_games', type=int, default=100, help='Number of games to play for architecture comparison')
    parser.add_argument('--plot_dir', type=str, default='./plots', help='Directory to save plots')
    # Add curriculum learning arguments
    parser.add_argument('--initial_complexity', type=float, default=0.1, help='Initial complexity for curriculum learning')
    parser.add_argument('--max_complexity', type=float, default=1.0, help='Maximum complexity for curriculum learning')
    parser.add_argument('--curriculum_steps', type=int, default=10, help='Number of steps in curriculum learning')
    parser.add_argument('--performance_threshold', type=float, default=0.6, help='Performance threshold for increasing complexity')
    
    args = parser.parse_args()

    logger = setup_logger('alphazero', 'alphazero_training.log')
    writer = SummaryWriter(log_dir=args.tensorboard_dir)

    logger.info('Starting AlphaZero training', extra={'action': 'training_start'})

    g = Game()
    game_phase_handler = GamePhaseHandler()

    architectures = {
        'original': AlphaZeroNet,
        'resnet': ResNetAlphaZero,
        'transformer': TransformerAlphaZero
    }

    nnets = {name: NNetWrapper(g, arch, game_phase_handler) for name, arch in architectures.items()}
    
    logger.info(f'Using {args.architecture} as the primary architecture')
    nnet = nnets[args.architecture]

    data_handler = DataHandler(args.data_dir)
    evaluator = ModelEvaluator(g, num_games=args.evaluation_games)
    
    # Initialize curriculum learning
    curriculum = CurriculumLearning(initial_complexity=args.initial_complexity,
                                    max_complexity=args.max_complexity,
                                    steps=args.curriculum_steps)
    
    curriculum_visualizer = CurriculumVisualizer()
    
    c = Coach(g, nnet, args, curriculum, game_phase_handler)
    analysis_toolkit = AnalysisToolkit(g, c.mcts)

    # Initialize ArchitectureVisualizer
    visualizer = ArchitectureVisualizer(architectures.keys(), args.plot_dir)

    # Load checkpoints if they exist
    for arch_name, arch_nnet in nnets.items():
        checkpoint_path = os.path.join(args.checkpoint, f'{arch_name}_latest.pth.tar')
        if os.path.exists(checkpoint_path):
            logger.info(f'Loading checkpoint for {arch_name} architecture')
            arch_nnet.load_checkpoint(args.checkpoint, f'{arch_name}_latest.pth.tar')

    logger.info('Starting the learning process 🎉', extra={'action': 'learning_start'})
    best_performance = float('-inf')
    early_stopping_counter = 0

    loss_data = {'total': [], 'policy': [], 'value': []}

    for i in range(1, args.numIters + 1):
        logger.info(f'Starting Iter #{i} ...', extra={'action': 'iteration_start', 'iteration': i})
        
        train_examples = c.executeEpisode()
        loss, pi_loss, v_loss = c.learn(train_examples)
        
        writer.add_scalar('Loss/total', loss, i)
        writer.add_scalar('Loss/policy', pi_loss, i)
        writer.add_scalar('Loss/value', v_loss, i)
        writer.add_scalar('Curriculum/complexity', curriculum.get_current_complexity(), i)

        loss_data['total'].append(loss)
        loss_data['policy'].append(pi_loss)
        loss_data['value'].append(v_loss)
        
        # Evaluate performance
        performance = evaluator.evaluate(nnet)
        curriculum.update_performance(performance)
        curriculum_visualizer.update(curriculum.get_current_complexity(), performance)
        
        logger.info(f'Iteration {i} - Loss: {loss:.4f}, Policy Loss: {pi_loss:.4f}, Value Loss: {v_loss:.4f}, '
                    f'Complexity: {curriculum.get_current_complexity():.2f}, Performance: {performance:.2f}',
                    extra={'action': 'training_metrics', 'iteration': i, 'loss': loss, 'pi_loss': pi_loss,
                           'v_loss': v_loss, 'complexity': curriculum.get_current_complexity(),
                           'performance': performance})
        
        if curriculum.should_increase_complexity():
            curriculum.step()
            logger.info(f'Increasing curriculum complexity to {curriculum.get_current_complexity():.2f}')
        
        if i % args.compare_every == 0:
            logger.info(f'Comparing architectures at iteration {i}')
            results, best_arch = compare_architectures(g.board_size, g.action_size, num_games=args.comparison_games)
            
            visualizer.update_data(i, results)
            performance_plot = visualizer.plot_performance()
            loss_plot = visualizer.plot_loss(loss_data)
            
            curriculum_visualizer.plot_progress(save_path=os.path.join(args.plot_dir, f'curriculum_progress_{i}.png'))
            curriculum_visualizer.plot_complexity_vs_performance(save_path=os.path.join(args.plot_dir, f'complexity_vs_performance_{i}.png'))

    # Final plots
    curriculum_visualizer.plot_progress(save_path=os.path.join(args.plot_dir, 'final_curriculum_progress.png'))
    curriculum_visualizer.plot_complexity_vs_performance(save_path=os.path.join(args.plot_dir, 'final_complexity_vs_performance.png'))

    logger.info('Training completed 🎉', extra={'action': 'training_complete'})

if __name__ == "__main__":
    main()
