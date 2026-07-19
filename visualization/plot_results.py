"""
Visualization Script
====================
Plot training metrics to see how the agent learned.

This creates graphs showing:
- Score over time
- Average score over time
- Loss over time
- Exploration rate (epsilon) over time
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_training_metrics(metrics_path, save_path=None):
    """
    Plot training metrics from a JSON file.
    
    Args:
        metrics_path: Path to metrics JSON file
        save_path: Where to save the plot (if None, just show it)
    """
    # Load metrics
    if not os.path.exists(metrics_path):
        print(f"ERROR: Metrics file not found: {metrics_path}")
        return
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    episodes = metrics['episodes']
    scores = metrics['scores']
    avg_scores = metrics['avg_scores']
    losses = metrics['losses']
    epsilons = metrics['epsilons']
    
    # Set style
    sns.set_style("darkgrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Score per episode
    axes[0, 0].plot(episodes, scores, alpha=0.3, color='blue', label='Score')
    axes[0, 0].plot(episodes, avg_scores, color='red', linewidth=2, label='Average (last 100)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Score Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average score (smoothed)
    axes[0, 1].plot(episodes, avg_scores, color='green', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Score (last 100 episodes)')
    axes[0, 1].set_title('Learning Progress (Smoothed)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Loss over time
    if losses and any(l > 0 for l in losses):
        # Only plot non-zero losses
        non_zero_losses = [l for l in losses if l > 0]
        non_zero_episodes = [episodes[i] for i, l in enumerate(losses) if l > 0]
        axes[1, 0].plot(non_zero_episodes, non_zero_losses, color='orange', linewidth=1)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_yscale('log')  # Log scale for better visualization
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No loss data available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Training Loss')
    
    # Plot 4: Epsilon (exploration rate)
    axes[1, 1].plot(episodes, epsilons, color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon (Exploration Rate)')
    axes[1, 1].set_title('Exploration vs Exploitation')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("TRAINING STATISTICS")
    print("=" * 60)
    print(f"Total episodes: {len(episodes)}")
    print(f"Best score: {max(scores)}")
    print(f"Average score (last 100): {avg_scores[-1]:.2f}")
    print(f"Average score (all): {sum(scores) / len(scores):.2f}")
    print(f"Final epsilon: {epsilons[-1]:.4f}")
    print("=" * 60)


def compare_models(model_paths, labels=None):
    """
    Compare multiple training runs.
    
    Args:
        model_paths: List of paths to metrics files
        labels: List of labels for each model (optional)
    """
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(model_paths))]
    
    plt.figure(figsize=(12, 6))
    
    for path, label in zip(model_paths, labels):
        if os.path.exists(path):
            with open(path, 'r') as f:
                metrics = json.load(f)
            
            episodes = metrics['episodes']
            avg_scores = metrics['avg_scores']
            
            plt.plot(episodes, avg_scores, label=label, linewidth=2)
        else:
            print(f"Warning: {path} not found")
    
    plt.xlabel('Episode')
    plt.ylabel('Average Score (last 100 episodes)')
    plt.title('Model Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Default metrics path
    default_metrics = "logs/flappy_bird_dqn_final_metrics.json"
    
    if len(sys.argv) > 1:
        metrics_path = sys.argv[1]
    else:
        metrics_path = default_metrics
    
    if not os.path.exists(metrics_path):
        print("Metrics file not found!")
        print(f"Looking for: {metrics_path}")
        print("\nPlease train a model first by running:")
        print("  python training/train.py")
        print("\nOr specify a different metrics path:")
        print("  python visualization/plot_results.py logs/your_metrics.json")
    else:
        # Plot and save
        save_path = metrics_path.replace('.json', '_plot.png')
        plot_training_metrics(metrics_path, save_path=save_path)
        
        # Also show interactively
        plot_training_metrics(metrics_path, save_path=None)

