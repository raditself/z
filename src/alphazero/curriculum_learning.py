
import numpy as np
from collections import deque

class Task:
    def __init__(self, difficulty, game_state):
        self.difficulty = difficulty
        self.game_state = game_state

class CurriculumLearner:
    def __init__(self, game, initial_tasks=None, buffer_size=1000):
        self.game = game
        self.task_buffer = deque(maxlen=buffer_size)
        self.performance_history = {}
        
        if initial_tasks:
            self.task_buffer.extend(initial_tasks)
        else:
            self.generate_initial_tasks()

    def generate_initial_tasks(self, num_tasks=100):
        for _ in range(num_tasks):
            difficulty = np.random.uniform(0, 1)
            game_state = self.game.generate_random_state(difficulty)
            task = Task(difficulty, game_state)
            self.task_buffer.append(task)

    def get_next_task(self):
        if np.random.random() < 0.1:  # Exploration: 10% chance to get a random task
            return np.random.choice(self.task_buffer)
        else:  # Exploitation: get a task based on the agent's current skill level
            return self.select_task_by_difficulty()

    def select_task_by_difficulty(self):
        agent_skill = self.estimate_agent_skill()
        tasks = sorted(self.task_buffer, key=lambda x: abs(x.difficulty - agent_skill))
        return tasks[0]  # Return the task with difficulty closest to the agent's skill

    def estimate_agent_skill(self):
        if not self.performance_history:
            return 0.5  # Default to medium difficulty if no history
        return np.mean(list(self.performance_history.values()))

    def update_curriculum(self, task, performance):
        self.performance_history[task] = performance
        
        # Generate new tasks based on performance
        if performance > 0.8:  # If performance is good, generate harder tasks
            new_difficulty = min(task.difficulty * 1.1, 1.0)
        elif performance < 0.2:  # If performance is poor, generate easier tasks
            new_difficulty = max(task.difficulty * 0.9, 0.0)
        else:
            new_difficulty = task.difficulty
        
        new_game_state = self.game.generate_random_state(new_difficulty)
        new_task = Task(new_difficulty, new_game_state)
        self.task_buffer.append(new_task)

    def train(self, agent, num_episodes=1000):
        for episode in range(num_episodes):
            task = self.get_next_task()
            performance = agent.train_on_task(task.game_state)
            self.update_curriculum(task, performance)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Agent Skill: {self.estimate_agent_skill():.2f}")

if __name__ == "__main__":
    from src.games.chess import Chess
    from src.alphazero.advanced_alphazero import AdvancedAlphaZero

    game = Chess()
    agent = AdvancedAlphaZero(game)
    curriculum_learner = CurriculumLearner(game)
    
    curriculum_learner.train(agent)
