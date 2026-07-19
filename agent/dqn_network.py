"""
Deep Q-Network (DQN) Neural Network
====================================
This is the "brain" of our AI agent.

The neural network takes the game state as input and outputs
Q-values for each possible action. Higher Q-value = better action.

Architecture:
    Input (5 values) → Hidden Layer 1 (128 neurons) → Hidden Layer 2 (128 neurons) → Output (2 values)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network: A neural network that learns to predict Q-values.
    
    Q-value = "How good is this action in this state?"
    The network learns to predict: Q(state, action)
    
    Input: Game state (5 numbers)
    Output: Q-values for each action (2 numbers: [don't jump, jump])
    """
    
    def __init__(self, input_size=5, hidden_size=128, output_size=2):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of state features (5: bird_y, velocity, pipe_dist, gap_top, gap_bottom)
            hidden_size: Number of neurons in hidden layers (128)
            output_size: Number of actions (2: don't jump, jump)
        """
        super(DQN, self).__init__()
        
        # Layer 1: Input → Hidden Layer 1
        # Takes 5 inputs, outputs 128 values
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Layer 2: Hidden Layer 1 → Hidden Layer 2
        # Takes 128 inputs, outputs 128 values
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Layer 3: Hidden Layer 2 → Output
        # Takes 128 inputs, outputs 2 values (Q-values for each action)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # That's it! This is our simple but powerful brain.
    
    def forward(self, x):
        """
        Forward pass: Process input through the network.
        
        This is how the network "thinks":
        1. Input state goes through first layer
        2. Apply ReLU activation (makes it non-linear)
        3. Goes through second layer
        4. Apply ReLU activation
        5. Goes through output layer
        6. Return Q-values
        
        Args:
            x: Input state (can be single state or batch of states)
            
        Returns:
            Q-values for each action
        """
        # Layer 1: Input → Hidden 1
        x = self.fc1(x)
        x = F.relu(x)  # ReLU activation: max(0, x) - makes network non-linear
        
        # Layer 2: Hidden 1 → Hidden 2
        x = self.fc2(x)
        x = F.relu(x)  # ReLU activation again
        
        # Layer 3: Hidden 2 → Output
        x = self.fc3(x)
        # No activation on output - we want raw Q-values
        
        return x
    
    def predict(self, state):
        """
        Predict Q-values for a given state.
        This is a convenience function for single predictions.
        
        Args:
            state: Game state (numpy array or list)
            
        Returns:
            Q-values as numpy array
        """
        # Convert to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Add batch dimension if needed (neural networks expect batches)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Set network to evaluation mode (no gradient computation)
        self.eval()
        with torch.no_grad():  # Don't track gradients for prediction
            q_values = self.forward(state)
        
        # Return as numpy array
        return q_values.cpu().numpy()[0]


# Example usage and explanation
if __name__ == "__main__":
    print("Creating a DQN network...")
    print("\nNetwork Architecture:")
    print("Input (5) → Hidden 1 (128) → Hidden 2 (128) → Output (2)")
    print("\nInput features:")
    print("  [0] Bird Y position")
    print("  [1] Bird velocity")
    print("  [2] Distance to next pipe")
    print("  [3] Top of gap")
    print("  [4] Bottom of gap")
    print("\nOutput:")
    print("  [0] Q-value for 'don't jump'")
    print("  [1] Q-value for 'jump'")
    
    # Create network
    network = DQN()
    
    # Create a sample state
    sample_state = torch.FloatTensor([0.5, 0.0, 0.3, 0.4, 0.6])
    print(f"\nSample state: {sample_state}")
    
    # Get Q-values
    q_values = network.predict(sample_state)
    print(f"Q-values: {q_values}")
    print(f"Best action: {'Jump' if q_values[1] > q_values[0] else 'Don\'t jump'}")
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

