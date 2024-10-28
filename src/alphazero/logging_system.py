

import logging
import json
import os
from functools import wraps

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        return json.dumps(log_record)

class EnvironmentFilter(logging.Filter):
    def filter(self, record):
        record.node_name = os.environ.get('NODE_NAME', 'unknown')
        record.pod_name = os.environ.get('POD_NAME', 'unknown')
        return True

def setup_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = JSONFormatter()
    env_filter = EnvironmentFilter()
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addFilter(env_filter)
    logger.addHandler(console_handler)
    
    return logger

def log_decorator(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Starting {func.__name__}", extra={'action': func.__name__, 'status': 'start'})
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed {func.__name__}", extra={'action': func.__name__, 'status': 'success'})
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", extra={'action': func.__name__, 'status': 'error', 'error_info': str(e)})
                raise
        return wrapper
    return decorator

# Example usage
logger = setup_logger('alphazero', 'alphazero_training.log')

@log_decorator(logger)
def train_iteration(iteration):
    # Simulating a training iteration
    logger.info(f"Training iteration {iteration}", extra={'iteration': iteration})
    # ... training code ...

if __name__ == '__main__':
    for i in range(5):
        train_iteration(i)

