"""
Training Script
===============
This script trains the DQN agent to play Flappy Bird.

The training process:
1. Create game environment
2. Create AI agent
3. Play many games (episodes)
4. Agent learns from experiences
5. Save trained model
6. Track and save metrics
"""

import os
import sys
import numpy as np
import json
from datetime import datetime

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.flappy_bird import FlappyBird
from agent.dqn_agent import DQNAgent


def train_agent(
    episodes=2000,
    max_steps=10000,
    render=False,
    save_interval=100,
    model_name="flappy_bird_dqn"
):
    """
    Train the DQN agent to play Flappy Bird.
    
    Args:
        episodes: Number of games to play (2000)
        max_steps: Maximum steps per episode (10000)
        render: Whether to show the game visually (False = faster training)
        save_interval: Save model every N episodes (100)
        model_name: Name for saved model
    """
    print("=" * 60)
    print("FLAPPY BIRD DQN TRAINING")
    print("=" * 60)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create game environment
    print("\nCreating game environment...")
    env = FlappyBird(render=render)
    
    # Create AI agent
    print("Creating DQN agent...")
    agent = DQNAgent(
        state_size=5,
        action_size=2,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        target_update=1000
    )
    
    # Training metrics
    scores = []  # Score for each episode
    avg_scores = []  # Average of last 100 episodes
    losses = []  # Loss values
    epsilons = []  # Exploration rates
    
    print(f"\nStarting training for {episodes} episodes...")
    print("This will take a while. The agent starts random and learns over time.")
    print("-" * 60)
    
    # Training loop
    for episode in range(episodes):
        # Reset game
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_losses = []
        
        # Play one episode (one game)
        while steps < max_steps:
            # Agent chooses action
            action = agent.act(state, training=True)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience in memory
            agent.remember(state, action, reward, next_state, done)
            
            # Learn from past experiences
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render if enabled (slows down training significantly)
            if render:
                env.render_frame()
            
            # Break if game over
            if done:
                break
        
        # Record metrics
        score = info['score']
        scores.append(score)
        epsilons.append(agent.epsilon)
        
        # Calculate average of last 100 episodes
        if len(scores) >= 100:
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
        else:
            avg_score = np.mean(scores)
            avg_scores.append(avg_score)
        
        # Average loss for this episode
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            losses.append(avg_loss)
        else:
            losses.append(0)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Score: {score:4d} | "
                  f"Avg Score (last 100): {avg_score:6.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}" if episode_losses else f"Loss: 0.0000")
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            model_path = f"models/{model_name}_episode_{episode + 1}.pth"
            agent.save(model_path)
            
            # Save training metrics
            metrics = {
                'episodes': list(range(1, episode + 2)),
                'scores': scores,
                'avg_scores': avg_scores,
                'losses': losses,
                'epsilons': epsilons
            }
            
            metrics_path = f"logs/{model_name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            
            print(f"  Model and metrics saved!")
    
    # Final save
    print("\n" + "-" * 60)
    print("Training completed!")
    final_model_path = f"models/{model_name}_final.pth"
    agent.save(final_model_path)
    
    # Save final metrics
    metrics = {
        'episodes': list(range(1, episodes + 1)),
        'scores': scores,
        'avg_scores': avg_scores,
        'losses': losses,
        'epsilons': epsilons,
        'training_date': datetime.now().isoformat(),
        'total_episodes': episodes
    }
    
    metrics_path = f"logs/{model_name}_final_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Metrics saved to: {metrics_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {episodes}")
    print(f"Best score: {max(scores)}")
    print(f"Average score (last 100): {avg_scores[-1]:.2f}")
    print(f"Average score (all): {np.mean(scores):.2f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    # You can modify these parameters to experiment!
    
    # For quick testing (fast but won't learn much)
    # train_agent(episodes=100, render=False)
    
    # For actual training (takes time but learns well)
    train_agent(
        episodes=2000,      # Number of games to play
        max_steps=10000,    # Max steps per game
        render=False,       # Set to True to watch training (much slower!)
        save_interval=100,  # Save every 100 episodes
        model_name="flappy_bird_dqn"
    )

