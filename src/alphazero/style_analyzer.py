
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class StyleAnalyzer:
    def __init__(self):
        self.move_frequencies = defaultdict(lambda: defaultdict(int))
        self.positional_preferences = defaultdict(lambda: defaultdict(float))
        self.risk_taking = []

    def update(self, game_state, move, policy, value):
        # Update move frequencies
        self.move_frequencies[str(game_state)][move] += 1

        # Update positional preferences
        for i, prob in enumerate(policy):
            self.positional_preferences[str(game_state)][i] += prob

        # Update risk-taking measure
        self.risk_taking.append(np.std(policy))

    def analyze_style(self, num_iterations):
        avg_risk = np.mean(self.risk_taking)
        move_diversity = np.mean([len(moves) for moves in self.move_frequencies.values()])
        positional_bias = np.mean([np.std(list(prefs.values())) for prefs in self.positional_preferences.values()])

        return {
            'avg_risk': avg_risk,
            'move_diversity': move_diversity,
            'positional_bias': positional_bias
        }

    def plot_style_evolution(self, style_metrics):
        iterations = list(range(1, len(style_metrics) + 1))
        metrics = ['avg_risk', 'move_diversity', 'positional_bias']

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig.suptitle("AlphaZero Style Evolution")

        for i, metric in enumerate(metrics):
            values = [m[metric] for m in style_metrics]
            axs[i].plot(iterations, values)
            axs[i].set_xlabel('Training Iterations')
            axs[i].set_ylabel(metric.replace('_', ' ').title())
            axs[i].grid(True)

        plt.tight_layout()
        plt.savefig('alphazero_style_evolution.png')
        plt.close()

    def generate_style_report(self, num_iterations, style_metrics):
        current_style = self.analyze_style(num_iterations)
        report = f"AlphaZero Style Analysis after {num_iterations} iterations:\n"
        report += f"Average Risk-Taking: {current_style['avg_risk']:.2f}\n"
        report += f"Move Diversity: {current_style['move_diversity']:.2f}\n"
        report += f"Positional Bias: {current_style['positional_bias']:.2f}\n\n"

        report += "Style Evolution:\n"
        for metric in ['avg_risk', 'move_diversity', 'positional_bias']:
            initial = style_metrics[0][metric]
            final = style_metrics[-1][metric]
            change = ((final - initial) / initial) * 100
            report += f"{metric.replace('_', ' ').title()}: {change:+.2f}% change\n"

        return report
