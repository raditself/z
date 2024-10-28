
import numpy as np
import os
import zlib
from typing import List, Tuple

class DataHandler:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.game_states_file = os.path.join(data_dir, 'game_states.npy')
        self.policy_targets_file = os.path.join(data_dir, 'policy_targets.npy')
        self.value_targets_file = os.path.join(data_dir, 'value_targets.npy')
        self.metadata_file = os.path.join(data_dir, 'metadata.npy')

    def save_game_data(self, game_states: List[np.ndarray], policy_targets: List[np.ndarray], value_targets: List[float]):
        game_states = np.array(game_states)
        policy_targets = np.array(policy_targets)
        value_targets = np.array(value_targets)

        if os.path.exists(self.game_states_file):
            with np.load(self.game_states_file, mmap_mode='r+') as f:
                game_states = np.concatenate((f, game_states))
        np.save(self.game_states_file, game_states)

        if os.path.exists(self.policy_targets_file):
            with np.load(self.policy_targets_file, mmap_mode='r+') as f:
                policy_targets = np.concatenate((f, policy_targets))
        np.save(self.policy_targets_file, policy_targets)

        if os.path.exists(self.value_targets_file):
            with np.load(self.value_targets_file, mmap_mode='r+') as f:
                value_targets = np.concatenate((f, value_targets))
        np.save(self.value_targets_file, value_targets)

        metadata = np.array([len(game_states)])
        np.save(self.metadata_file, metadata)

    def load_game_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        game_states = np.load(self.game_states_file, mmap_mode='r')
        policy_targets = np.load(self.policy_targets_file, mmap_mode='r')
        value_targets = np.load(self.value_targets_file, mmap_mode='r')
        return game_states, policy_targets, value_targets

    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        metadata = np.load(self.metadata_file)
        total_samples = metadata[0]
        
        indices = np.random.choice(total_samples, batch_size, replace=False)
        
        game_states, policy_targets, value_targets = self.load_game_data()
        
        return game_states[indices], policy_targets[indices], value_targets[indices]

    def compress_data(self):
        for file in [self.game_states_file, self.policy_targets_file, self.value_targets_file]:
            with open(file, 'rb') as f_in:
                with open(f'{file}.gz', 'wb') as f_out:
                    f_out.write(zlib.compress(f_in.read()))
            os.remove(file)

    def decompress_data(self):
        for file in [f'{self.game_states_file}.gz', f'{self.policy_targets_file}.gz', f'{self.value_targets_file}.gz']:
            with open(file, 'rb') as f_in:
                with open(file[:-3], 'wb') as f_out:
                    f_out.write(zlib.decompress(f_in.read()))
            os.remove(file)
