import time

class ChessTimer:
    def __init__(self, initial_time, increment):
        self.initial_time = initial_time
        self.increment = increment
        self.white_time = initial_time
        self.black_time = initial_time
        self.current_player = 'white'
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if self.current_player == 'white':
                self.white_time -= elapsed
                self.white_time += self.increment
            else:
                self.black_time -= elapsed
                self.black_time += self.increment
            self.start_time = None

    def switch_player(self):
        self.stop()
        self.current_player = 'black' if self.current_player == 'white' else 'white'
        self.start()

    def get_time(self, color):
        if self.start_time is not None and color == self.current_player:
            elapsed = time.time() - self.start_time
            return max(0, (self.white_time if color == 'white' else self.black_time) - elapsed)
        return self.white_time if color == 'white' else self.black_time

    def is_flag_fallen(self, color):
        return self.get_time(color) <= 0
