
import logging
import os
from torch.utils.tensorboard import SummaryWriter

class AlphaZeroLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger = logging.getLogger('AlphaZero')
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(os.path.join(log_dir, 'alphazero.log'))
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))

    def log_info(self, message):
        self.logger.info(message)

    def log_error(self, message):
        self.logger.error(message)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def close(self):
        self.writer.close()
