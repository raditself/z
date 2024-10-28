
import matplotlib.pyplot as plt
import numpy as np

class CurriculumVisualizer:
    def __init__(self):
        self.complexity_history = []
        self.performance_history = []

    def update(self, complexity, performance):
        self.complexity_history.append(complexity)
        self.performance_history.append(performance)

    def plot_progress(self, save_path=None):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.complexity_history)
        plt.title('Curriculum Complexity Over Time')
        plt.ylabel('Complexity')

        plt.subplot(2, 1, 2)
        plt.plot(self.performance_history)
        plt.title('Performance Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Performance')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_complexity_vs_performance(self, save_path=None):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.complexity_history, self.performance_history)
        plt.title('Complexity vs Performance')
        plt.xlabel('Complexity')
        plt.ylabel('Performance')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
