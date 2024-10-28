
import logging
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

class AlphaZeroLogger:
    def __init__(self, log_dir):
        self.logger = logging.getLogger('AlphaZero')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"{log_dir}/alphazero.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.writer = SummaryWriter(log_dir)
        self.start_time = time.time()

    def log_iteration(self, iteration, loss, policy_loss, value_loss, game_length):
        self.logger.info(f"Iteration {iteration}: Loss: {loss:.4f}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Game Length: {game_length}")
        self.writer.add_scalar('Loss/total', loss, iteration)
        self.writer.add_scalar('Loss/policy', policy_loss, iteration)
        self.writer.add_scalar('Loss/value', value_loss, iteration)
        self.writer.add_scalar('Game/length', game_length, iteration)

    def log_evaluation(self, iteration, win_rate):
        self.logger.info(f"Iteration {iteration}: Win Rate: {win_rate:.2f}")
        self.writer.add_scalar('Evaluation/win_rate', win_rate, iteration)

    def log_mcts_stats(self, iteration, num_nodes, max_depth):
        self.writer.add_scalar('MCTS/num_nodes', num_nodes, iteration)
        self.writer.add_scalar('MCTS/max_depth', max_depth, iteration)

    def log_time(self, iteration):
        elapsed_time = time.time() - self.start_time
        self.writer.add_scalar('Time/elapsed_hours', elapsed_time / 3600, iteration)

    def close(self):
        self.writer.close()

def plot_game_length_distribution(game_lengths, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(game_lengths, kde=True)
    plt.title('Distribution of Game Lengths')
    plt.xlabel('Number of Moves')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def plot_action_heatmap(action_frequencies, board_size, save_path):
    plt.figure(figsize=(10, 10))
    sns.heatmap(action_frequencies.reshape(board_size, board_size), annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Heatmap of Action Frequencies')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    logger = AlphaZeroLogger('logs')
    
    # Simulating training progress
    for i in range(100):
        loss = 1.0 / (i + 1)
        policy_loss = loss * 0.6
        value_loss = loss * 0.4
        game_length = 50 + i // 2
        logger.log_iteration(i, loss, policy_loss, value_loss, game_length)
        
        if i % 10 == 0:
            win_rate = 0.5 + i / 200
            logger.log_evaluation(i, win_rate)
        
        logger.log_mcts_stats(i, 1000 + i * 10, 20 + i // 5)
        logger.log_time(i)
    
    logger.close()
    
    # Example of plotting game length distribution
    game_lengths = [50 + i // 2 for i in range(100)]
    plot_game_length_distribution(game_lengths, 'logs/game_length_distribution.png')
    
    # Example of plotting action heatmap
    action_frequencies = np.random.randint(0, 100, size=(8, 8))
    plot_action_heatmap(action_frequencies, 8, 'logs/action_heatmap.png')

    print("Logging and visualization examples created.")
