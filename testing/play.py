"""
Play Script
===========
Watch your trained AI agent play Flappy Bird!

This script loads a trained model and lets you watch the AI play.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.flappy_bird import FlappyBird
from agent.dqn_agent import DQNAgent
import pygame


def play_game(model_path, num_games=5):
    """
    Watch the trained agent play Flappy Bird.
    
    Args:
        model_path: Path to the trained model file
        num_games: Number of games to watch
    """
    print("=" * 60)
    print("WATCHING AI PLAY FLAPPY BIRD")
    print("=" * 60)
    
    # Create game environment (with rendering enabled)
    print("\nCreating game environment...")
    env = FlappyBird(render=True)
    
    # Create agent
    print("Creating DQN agent...")
    agent = DQNAgent()
    
    # Load trained model
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found: {model_path}")
        print("\nAvailable models:")
        if os.path.exists("models"):
            for file in os.listdir("models"):
                if file.endswith(".pth"):
                    print(f"  - models/{file}")
        return
    
    print(f"Loading model from: {model_path}")
    agent.load(model_path)
    agent.epsilon = 0  # No exploration - always use learned knowledge
    
    print(f"\nWatching {num_games} games...")
    print("Close the window or press ESC to stop early.")
    print("-" * 60)
    
    scores = []
    
    for game in range(num_games):
        # Reset game
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nGame {game + 1}/{num_games} starting...")
        
        running = True
        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
            
            if not running:
                break
            
            # Agent chooses action (no exploration - always exploit)
            action = agent.act(state, training=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render game
            env.render_frame()
            
            # Small delay to make it watchable
            pygame.time.delay(10)
            
            # Check if game over
            if done:
                score = info['score']
                scores.append(score)
                print(f"  Game Over! Score: {score}")
                
                # Wait a bit before next game
                pygame.time.wait(2000)
                break
        
        if not running:
            break
    
    env.close()
    
    # Print summary
    if scores:
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Games played: {len(scores)}")
        print(f"Best score: {max(scores)}")
        print(f"Average score: {sum(scores) / len(scores):.2f}")
        print(f"All scores: {scores}")
        print("=" * 60)


if __name__ == "__main__":
    # Default model path
    default_model = "models/flappy_bird_dqn_final.pth"
    
    # Check if model exists
    if not os.path.exists(default_model):
        print("Trained model not found!")
        print(f"Looking for: {default_model}")
        print("\nPlease train a model first by running:")
        print("  python training/train.py")
        print("\nOr specify a different model path:")
        print("  python testing/play.py models/your_model.pth")
    else:
        # Play with default model
        play_game(default_model, num_games=5)
        
        # Or specify a different model:
        # play_game("models/flappy_bird_dqn_episode_1000.pth", num_games=10)

