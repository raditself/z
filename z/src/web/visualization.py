
import matplotlib.pyplot as plt
import io
import base64
from ..alphazero.logging_system import AdvancedLogger

def generate_training_progress_graph():
    logger = AdvancedLogger()
    data = logger.get_training_progress(limit=100)
    
    iterations = [row[2] for row in data]
    losses = [row[3] for row in data]
    accuracies = [row[4] for row in data]
    elo_ratings = [row[5] for row in data]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    ax1.plot(iterations, losses, label='Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(iterations, accuracies, label='Accuracy')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    ax3.plot(iterations, elo_ratings, label='Elo Rating')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Elo Rating')
    ax3.legend()

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f'data:image/png;base64,{graph_url}'

