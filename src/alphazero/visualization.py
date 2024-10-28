
# Content of visualization.py
import matplotlib.pyplot as plt
import os

class ArchitectureVisualizer:
    def __init__(self, architectures, save_dir):
        self.architectures = architectures
        self.save_dir = save_dir
        self.performance_data = {arch: [] for arch in architectures}
        self.iterations = []

    def update_data(self, iteration, performance_dict):
        self.iterations.append(iteration)
        for arch, perf in performance_dict.items():
            self.performance_data[arch].append(perf)

    def plot_performance(self):
        plt.figure(figsize=(12, 8))
        for arch in self.architectures:
            plt.plot(self.iterations, self.performance_data[arch], label=arch)

        plt.xlabel('Iteration')
        plt.ylabel('Performance')
        plt.title('Architecture Performance over Time')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(self.save_dir, 'architecture_performance.png')
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    def plot_loss(self, loss_data):
        plt.figure(figsize=(12, 8))
        for loss_type, values in loss_data.items():
            plt.plot(self.iterations, values, label=loss_type)

        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss over Time')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(self.save_dir, 'loss_over_time.png')
        plt.savefig(plot_path)
        plt.close()

        return plot_path

# Write the content to visualization.py
with open('/home/user/z/src/alphazero/visualization.py', 'w') as f:
    f.write(CODE)

print("visualization.py has been created with functions for plotting architecture performance and loss.")

# Verify the contents of the file
with open('/home/user/z/src/alphazero/visualization.py', 'r') as f:
    print(f.read())
