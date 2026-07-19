"""
Flappy Bird Game Environment
============================
This is the game world where our AI agent will learn to play.

The game environment provides:
- A visual game that the AI can interact with
- State information (what the AI "sees")
- Reward signals (what the AI "feels" - good or bad)
- Action interface (how the AI "acts")
"""

import pygame
import numpy as np
import random

# Game constants - these control the game's appearance and behavior
SCREEN_WIDTH = 400      # Width of game window
SCREEN_HEIGHT = 600     # Height of game window
GRAVITY = 0.5           # How fast bird falls (pixels per frame)
JUMP_STRENGTH = -8      # How high bird jumps (negative = upward)
PIPE_WIDTH = 60         # Width of pipes
PIPE_GAP = 150          # Gap between top and bottom pipes
PIPE_SPEED = 3          # How fast pipes move left
PIPE_SPACING = 200      # Distance between pipe pairs
GROUND_HEIGHT = 50      # Height of ground at bottom

# Colors (RGB values)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (255, 0, 0)
BLUE = (100, 150, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)


class FlappyBird:
    """
    The Flappy Bird game environment.
    
    This class creates and manages the game world. It's like a teacher
    that shows the AI what's happening and gives it feedback.
    """
    
    def __init__(self, render=True):
        """
        Initialize the game.
        
        Args:
            render: If True, shows the game visually. If False, runs headless (faster for training)
        """
        self.render = render
        
        # Initialize Pygame if we want to see the game
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - RL Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        # Reset game to initial state
        self.reset()
    
    def reset(self):
        """
        Reset the game to start a new episode.
        This is called at the beginning of each training episode.
        
        Returns:
            state: The initial game state (what the AI sees)
        """
        # Bird starting position (center-left of screen)
        self.bird_x = 50
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_velocity = 0  # Bird's vertical speed
        
        # Create first pipe pair
        self.pipes = []
        self._add_pipe()
        
        # Game state
        self.score = 0
        self.game_over = False
        self.passed_pipes = set()  # Track which pipes we've passed
        
        # Return the initial state
        return self._get_state()
    
    def _add_pipe(self):
        """
        Add a new pipe pair to the game.
        Pipes are added at the right edge of the screen.
        """
        # Random gap position (but not too high or too low)
        gap_y = random.randint(150, SCREEN_HEIGHT - GROUND_HEIGHT - 150)
        
        # Create top pipe (hanging down from top)
        top_pipe = {
            'x': SCREEN_WIDTH,
            'top': 0,
            'bottom': gap_y - PIPE_GAP // 2
        }
        
        # Create bottom pipe (coming up from bottom)
        bottom_pipe = {
            'x': SCREEN_WIDTH,
            'top': gap_y + PIPE_GAP // 2,
            'bottom': SCREEN_HEIGHT - GROUND_HEIGHT
        }
        
        self.pipes.append((top_pipe, bottom_pipe))
    
    def step(self, action):
        """
        Take one step in the game.
        
        This is the main function the AI calls repeatedly:
        1. AI decides on an action (jump or don't jump)
        2. We apply that action
        3. Game physics update
        4. We check for collisions
        5. We calculate reward
        6. We return new state and reward
        
        Args:
            action: 0 = don't jump, 1 = jump
            
        Returns:
            state: New game state
            reward: Reward for this action (+ = good, - = bad)
            done: True if game over, False if still playing
            info: Additional information (like score)
        """
        if self.game_over:
            return self._get_state(), 0, True, {'score': self.score}
        
        # Apply action: if action is 1, make bird jump
        if action == 1:
            self.bird_velocity = JUMP_STRENGTH
        
        # Update bird physics
        self.bird_velocity += GRAVITY  # Gravity pulls bird down
        self.bird_y += self.bird_velocity  # Update bird position
        
        # Move pipes to the left
        for top_pipe, bottom_pipe in self.pipes:
            top_pipe['x'] -= PIPE_SPEED
            bottom_pipe['x'] -= PIPE_SPEED
        
        # Add new pipe when needed
        if len(self.pipes) == 0 or self.pipes[-1][0]['x'] < SCREEN_WIDTH - PIPE_SPACING:
            self._add_pipe()
        
        # Remove pipes that are off screen
        self.pipes = [(top, bottom) for top, bottom in self.pipes if top['x'] > -PIPE_WIDTH]
        
        # Check for collisions
        reward = 0.1  # Small reward for staying alive
        done = False
        
        # Check collision with ground or ceiling
        if self.bird_y >= SCREEN_HEIGHT - GROUND_HEIGHT or self.bird_y <= 0:
            reward = -10  # Big penalty for dying
            done = True
            self.game_over = True
        
        # Check collision with pipes
        bird_rect = pygame.Rect(self.bird_x - 10, self.bird_y - 10, 20, 20)
        
        for i, (top_pipe, bottom_pipe) in enumerate(self.pipes):
            # Top pipe collision
            top_rect = pygame.Rect(
                top_pipe['x'], 
                top_pipe['top'], 
                PIPE_WIDTH, 
                top_pipe['bottom'] - top_pipe['top']
            )
            
            # Bottom pipe collision
            bottom_rect = pygame.Rect(
                bottom_pipe['x'],
                bottom_pipe['top'],
                PIPE_WIDTH,
                bottom_pipe['bottom'] - bottom_pipe['top']
            )
            
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                reward = -10  # Big penalty for hitting pipe
                done = True
                self.game_over = True
                break
            
            # Check if bird passed through pipe (scoring)
            # We check if pipe is behind bird and we haven't scored it yet
            if (top_pipe['x'] + PIPE_WIDTH < self.bird_x and 
                i not in self.passed_pipes):
                self.score += 1
                self.passed_pipes.add(i)
                reward = 10  # Big reward for passing a pipe!
        
        return self._get_state(), reward, done, {'score': self.score}
    
    def _get_state(self):
        """
        Get the current game state.
        
        This is what the AI "sees". We convert the game world into
        numbers that the neural network can understand.
        
        Returns:
            state: A numpy array with 5 values:
                [0] Bird's Y position (normalized)
                [1] Bird's velocity
                [2] Distance to next pipe (normalized)
                [3] Top of gap (normalized)
                [4] Bottom of gap (normalized)
        """
        # Normalize bird Y position (0 to 1)
        bird_y_norm = self.bird_y / SCREEN_HEIGHT
        
        # Find the next pipe (closest pipe in front of bird)
        next_pipe = None
        for top_pipe, bottom_pipe in self.pipes:
            if top_pipe['x'] + PIPE_WIDTH > self.bird_x:
                next_pipe = (top_pipe, bottom_pipe)
                break
        
        if next_pipe is None:
            # If no pipe found, use default values
            pipe_dist = 1.0
            gap_top = 0.5
            gap_bottom = 0.5
        else:
            top_pipe, bottom_pipe = next_pipe
            
            # Distance to pipe (normalized)
            pipe_dist = (top_pipe['x'] - self.bird_x) / SCREEN_WIDTH
            
            # Gap positions (normalized)
            gap_top = bottom_pipe['top'] / SCREEN_HEIGHT
            gap_bottom = (bottom_pipe['top'] + PIPE_GAP) / SCREEN_HEIGHT
        
        # Return state as numpy array
        state = np.array([
            bird_y_norm,      # [0] Bird position
            self.bird_velocity / 10,  # [1] Bird velocity (normalized)
            pipe_dist,        # [2] Distance to pipe
            gap_top,          # [3] Top of gap
            gap_bottom        # [4] Bottom of gap
        ], dtype=np.float32)
        
        return state
    
    def render_frame(self):
        """
        Draw the game on screen.
        This is only called when render=True.
        """
        if not self.render:
            return
        
        # Fill background with sky blue
        self.screen.fill(BLUE)
        
        # Draw pipes
        for top_pipe, bottom_pipe in self.pipes:
            # Top pipe (hanging down)
            pygame.draw.rect(
                self.screen, 
                GREEN,
                (top_pipe['x'], top_pipe['top'], PIPE_WIDTH, top_pipe['bottom'] - top_pipe['top'])
            )
            
            # Bottom pipe (coming up)
            pygame.draw.rect(
                self.screen,
                GREEN,
                (bottom_pipe['x'], bottom_pipe['top'], PIPE_WIDTH, bottom_pipe['bottom'] - bottom_pipe['top'])
            )
        
        # Draw bird (simple circle)
        pygame.draw.circle(self.screen, YELLOW, (int(self.bird_x), int(self.bird_y)), 15)
        # Draw bird's eye
        pygame.draw.circle(self.screen, BLACK, (int(self.bird_x) + 5, int(self.bird_y) - 3), 3)
        
        # Draw ground
        pygame.draw.rect(
            self.screen,
            ORANGE,
            (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT)
        )
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Draw game over message
        if self.game_over:
            game_over_text = self.font.render("GAME OVER", True, RED)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS
    
    def close(self):
        """Close the game window."""
        if self.render:
            pygame.quit()


# Test the game (you can run this file directly to see the game)
if __name__ == "__main__":
    print("Testing Flappy Bird game...")
    print("Click the window and press SPACE to make the bird jump!")
    
    game = FlappyBird(render=True)
    state = game.reset()
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Jump action
                    state, reward, done, info = game.step(1)
                else:
                    # Don't jump
                    state, reward, done, info = game.step(0)
        
        # Auto-play: don't jump (bird falls)
        state, reward, done, info = game.step(0)
        
        # Render
        game.render_frame()
        
        # Reset if game over
        if done:
            print(f"Game Over! Score: {info['score']}")
            pygame.time.wait(2000)  # Wait 2 seconds
            state = game.reset()
    
    game.close()

