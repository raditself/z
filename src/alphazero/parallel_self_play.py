
import multiprocessing as mp
from .alphazero import AlphaZero

def self_play_worker(game, args, return_queue):
    alphazero = AlphaZero(game, args)
    memory = alphazero.self_play()
    return_queue.put(memory)

class ParallelSelfPlay:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.num_processes = mp.cpu_count()

    def parallel_self_play(self, num_games):
        manager = mp.Manager()
        return_queue = manager.Queue()
        processes = []

        games_per_process = num_games // self.num_processes
        remaining_games = num_games % self.num_processes

        for i in range(self.num_processes):
            num_games_for_process = games_per_process + (1 if i < remaining_games else 0)
            p = mp.Process(target=self._run_self_play, args=(num_games_for_process, return_queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        all_memories = []
        while not return_queue.empty():
            all_memories.extend(return_queue.get())

        return all_memories

    def _run_self_play(self, num_games, return_queue):
        memories = []
        for _ in range(num_games):
            alphazero = AlphaZero(self.game, self.args)
            memories.extend(alphazero.self_play())
        return_queue.put(memories)
