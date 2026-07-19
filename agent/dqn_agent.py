"""
DQN Agent
=========
This is the decision-maker that uses the neural network to play the game.

The agent:
1. Chooses actions (exploration vs exploitation)
2. Remembers experiences (state, action, reward, next_state)
3. Learns from past experiences
4. Updates the neural network

Key concepts:
- Epsilon-Greedy: Balance between exploration and exploitation
- Experience Replay: Learn from random past experiences
- Target Network: Stable learning target
"""

import numpy as np
import random
import torch
import torch.optim as optim
from collections import deque
from agent.dqn_network import DQN


class DQNAgent:
    """
    Deep Q-Network Agent
    
    This agent learns to play Flappy Bird by:
    1. Observing the game state
    2. Choosing actions (jump or don't jump)
    3. Remembering what happened
    4. Learning from past experiences
    """
    
    def __init__(
        self,
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
    ):
        """
        Initialize the DQN Agent.
        
        Args:
            state_size: Number of state features (5)
            action_size: Number of actions (2: don't jump, jump)
            learning_rate: How fast the network learns (0.001)
            gamma: Discount factor - how much we value future rewards (0.99)
            epsilon: Initial exploration rate (1.0 = 100% random)
            epsilon_min: Minimum exploration rate (0.01 = 1% random)
            epsilon_decay: How fast epsilon decreases (0.995)
            memory_size: Size of experience replay buffer (10000)
            batch_size: How many experiences to learn from at once (32)
            target_update: How often to update target network (every 1000 steps)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0  # Track how many learning steps we've done
        
        # Create the main network (this one gets updated frequently)
        self.q_network = DQN(state_size, 128, action_size)
        
        # Create the target network (this one updates slowly for stability)
        self.target_network = DQN(state_size, 128, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Copy weights
        
        # Optimizer: Updates the network weights
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience Replay Memory
        # This stores past experiences: (state, action, reward, next_state, done)
        self.memory = deque(maxlen=memory_size)
        
        # Device: Use GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
        print(f"DQN Agent initialized on {self.device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Gamma (discount): {gamma}")
        print(f"  Epsilon (exploration): {epsilon}")
        print(f"  Memory size: {memory_size}")
        print(f"  Batch size: {batch_size}")
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in memory.
        
        This is like writing in a diary:
        "I was in state X, did action Y, got reward Z, ended up in state W, game over: True/False"
        
        Args:
            state: Current game state
            action: Action taken
            reward: Reward received
            next_state: State after taking action
            done: Whether game ended
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose an action using epsilon-greedy strategy.
        
        Epsilon-greedy means:
        - With probability epsilon: Explore (random action)
        - With probability (1-epsilon): Exploit (use learned knowledge)
        
        Args:
            state: Current game state
            training: If True, use epsilon-greedy. If False, always exploit.
            
        Returns:
            action: 0 (don't jump) or 1 (jump)
        """
        if training and random.random() <= self.epsilon:
            # Exploration: Random action
            return random.randrange(self.action_size)
        else:
            # Exploitation: Use neural network to choose best action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            self.q_network.train()
            
            # Return action with highest Q-value
            return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """
        Learn from past experiences.
        
        This is the core learning function:
        1. Sample random batch of experiences from memory
        2. Calculate what Q-values should be (targets)
        3. Calculate what Q-values currently are (predictions)
        4. Calculate loss (difference between target and prediction)
        5. Update network to minimize loss
        
        This is called periodically during training.
        """
        # Need enough experiences to learn from
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Separate the batch into components
        # Convert to numpy arrays first for better performance
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in batch])).to(self.device)
        
        # Get current Q-values (what the network currently predicts)
        # We only want Q-values for actions we actually took
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q-values from target network
        # We use target network for stability
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
        
        # Calculate target Q-values
        # If game over: target = reward (no future)
        # If not over: target = reward + gamma * max_future_q_value
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        target_q_values = target_q_values.unsqueeze(1)
        
        # Calculate loss (how wrong our predictions are)
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
        
        # Update network
        self.optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Calculate new gradients
        self.optimizer.step()  # Update weights
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"  Target network updated (step {self.update_counter})")
        
        # Decay epsilon (reduce exploration over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath):
        """
        Save the trained network to a file.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a trained network from a file.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        print(f"Model loaded from {filepath}")


# Test the agent
if __name__ == "__main__":
    print("Testing DQN Agent...")
    
    # Create agent
    agent = DQNAgent()
    
    # Create a sample state
    sample_state = np.array([0.5, 0.0, 0.3, 0.4, 0.6])
    
    # Test action selection
    print("\nTesting action selection:")
    for i in range(10):
        action = agent.act(sample_state, training=True)
        print(f"  Step {i+1}: Action = {action} (epsilon = {agent.epsilon:.3f})")
    
    # Test memory
    print("\nTesting experience storage:")
    for i in range(5):
        next_state = np.random.rand(5)
        agent.remember(sample_state, 1, 0.1, next_state, False)
    print(f"  Memory size: {len(agent.memory)}")
    
    # Test learning (need more experiences first)
    print("\nAdding more experiences for learning test...")
    for i in range(100):
        state = np.random.rand(5)
        action = random.randint(0, 1)
        reward = random.uniform(-1, 1)
        next_state = np.random.rand(5)
        done = random.choice([True, False])
        agent.remember(state, action, reward, next_state, done)
    
    print(f"  Memory size: {len(agent.memory)}")
    print("  Testing learning step...")
    loss = agent.replay()
    print(f"  Loss: {loss:.4f}")

