
import matplotlib.pyplot as plt
import networkx as nx

def visualize_move_explanations(explanations):
    """
    Visualize the AI's move explanations using a decision tree-like graph.
    """
    G = nx.DiGraph()
    
    # Add root node
    G.add_node("AI Decision", pos=(0, 0))
    
    if explanations['type'] == 'Opening Book':
        G.add_node("Opening Book", pos=(0, -1))
        G.add_edge("AI Decision", "Opening Book")
        G.add_node(f"Move: {explanations['move']}", pos=(0, -2))
        G.add_edge("Opening Book", f"Move: {explanations['move']}")
    
    elif explanations['type'] == 'Random Move':
        G.add_node("Random Selection", pos=(0, -1))
        G.add_edge("AI Decision", "Random Selection")
        G.add_node(f"Move: {explanations['move']}", pos=(0, -2))
        G.add_edge("Random Selection", f"Move: {explanations['move']}")
    
    elif explanations['type'] == 'MCTS Evaluation':
        G.add_node("MCTS Evaluation", pos=(0, -1))
        G.add_edge("AI Decision", "MCTS Evaluation")
        
        # Add best move
        G.add_node(f"Best Move: {explanations['move']}", pos=(-2, -2))
        G.add_edge("MCTS Evaluation", f"Best Move: {explanations['move']}")
        G.add_node(f"Probability: {explanations['probability']:.2f}", pos=(-2, -3))
        G.add_edge(f"Best Move: {explanations['move']}", f"Probability: {explanations['probability']:.2f}")
        
        # Add top alternatives
        for i, (move, prob) in enumerate(explanations['top_alternatives'][:3]):  # Limit to top 3 for clarity
            G.add_node(f"Alternative {i+1}: {move}", pos=(i-0.5, -2))
            G.add_edge("MCTS Evaluation", f"Alternative {i+1}: {move}")
            G.add_node(f"Probability: {prob:.2f}", pos=(i-0.5, -3))
            G.add_edge(f"Alternative {i+1}: {move}", f"Probability: {prob:.2f}")
    
    # Draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=8, font_weight='bold')
    nx.draw_networkx_labels(G, pos)
    plt.title("AI Decision Process")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_move_probabilities(move_evaluations):
    """
    Plot a bar chart of move probabilities.
    """
    moves = []
    probabilities = []
    for move, prob in move_evaluations.items():
        if move != 'best_move':
            moves.append(str(move))
            probabilities.append(prob)
    
    plt.figure(figsize=(12, 6))
    plt.bar(moves, probabilities)
    plt.title('Move Probabilities')
    plt.xlabel('Moves')
    plt.ylabel('Probability')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
