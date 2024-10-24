
import math
import random

class Player:
    def __init__(self, rating=1500, k_factor=32):
        self.rating = rating
        self.k_factor = k_factor

class RatingSystem:
    def __init__(self):
        self.players = {}

    def add_player(self, player_id, rating=1500):
        self.players[player_id] = Player(rating)

    def get_expected_score(self, player_a, player_b):
        return 1 / (1 + math.pow(10, (player_b.rating - player_a.rating) / 400))

    def update_ratings(self, player_a_id, player_b_id, score):
        player_a = self.players[player_a_id]
        player_b = self.players[player_b_id]

        expected_a = self.get_expected_score(player_a, player_b)
        expected_b = 1 - expected_a

        player_a.rating += player_a.k_factor * (score - expected_a)
        player_b.rating += player_b.k_factor * ((1 - score) - expected_b)

    def get_rating(self, player_id):
        return self.players[player_id].rating

class AIOpponentPool:
    def __init__(self, num_opponents=10, min_rating=1000, max_rating=2000):
        self.rating_system = RatingSystem()
        self.opponents = []
        for i in range(num_opponents):
            rating = random.randint(min_rating, max_rating)
            opponent_id = f"AI_Opponent_{i}"
            self.rating_system.add_player(opponent_id, rating)
            self.opponents.append(opponent_id)

    def select_opponent(self, player_rating):
        # Select an opponent with a rating close to the player's rating
        closest_opponent = min(self.opponents, key=lambda x: abs(self.rating_system.get_rating(x) - player_rating))
        return closest_opponent

    def update_ratings(self, player_id, opponent_id, score):
        self.rating_system.update_ratings(player_id, opponent_id, score)

    def get_rating(self, player_id):
        return self.rating_system.get_rating(player_id)

# Example usage
if __name__ == "__main__":
    pool = AIOpponentPool()
    player_id = "Human_Player"
    pool.rating_system.add_player(player_id)

    for _ in range(5):
        opponent = pool.select_opponent(pool.get_rating(player_id))
        print(f"Player rating: {pool.get_rating(player_id):.2f}")
        print(f"Selected opponent: {opponent}, rating: {pool.get_rating(opponent):.2f}")
        
        # Simulate a game result (0 for loss, 0.5 for draw, 1 for win)
        score = random.choice([0, 0.5, 1])
        pool.update_ratings(player_id, opponent, score)
        print(f"Game result: {'Win' if score == 1 else 'Draw' if score == 0.5 else 'Loss'}")
        print(f"Updated player rating: {pool.get_rating(player_id):.2f}")
        print()
