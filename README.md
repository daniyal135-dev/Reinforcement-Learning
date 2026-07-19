# 🐦 Flappy Bird Frenzy: A Reinforcement Learning Approach

A complete implementation of Flappy Bird game with Deep Q-Network (DQN) reinforcement learning agent that learns to play the game automatically.

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Understanding the Components](#understanding-the-components)
3. [Installation](#installation)
4. [How to Use](#how-to-use)
5. [Project Structure](#project-structure)
6. [How It Works](#how-it-works)
7. [Hyperparameters Explained](#hyperparameters-explained)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project demonstrates how an AI agent can learn to play Flappy Bird using **Deep Q-Network (DQN)**, a popular reinforcement learning algorithm. The agent starts by playing randomly and gradually learns to achieve high scores through trial and error.

### What You'll Learn:
- How reinforcement learning works
- How neural networks can learn game strategies
- How to implement DQN from scratch
- How to visualize and analyze AI learning

---

## 🧩 Understanding the Components

### 1. **Game Environment** (`game/flappy_bird.py`)
- The Flappy Bird game itself
- Provides state information to the AI
- Gives rewards based on actions
- Handles game physics and collisions

### 2. **Neural Network** (`agent/dqn_network.py`)
- The "brain" of the AI
- Takes game state as input
- Outputs Q-values (quality scores) for each action
- Learns through backpropagation

### 3. **DQN Agent** (`agent/dqn_agent.py`)
- Decision maker that uses the neural network
- Implements epsilon-greedy strategy (exploration vs exploitation)
- Stores experiences in memory
- Learns from past experiences

### 4. **Training Script** (`training/train.py`)
- Orchestrates the learning process
- Runs multiple episodes (games)
- Tracks performance metrics
- Saves trained models

### 5. **Testing Script** (`testing/play.py`)
- Loads trained models
- Lets you watch the AI play
- Shows performance statistics

### 6. **Visualization** (`visualization/plot_results.py`)
- Creates graphs of training progress
- Shows score, loss, and exploration rate over time

---

## 🚀 Installation

### Step 1: Install Python
Make sure you have Python 3.8 or higher installed.

### Step 2: Create Virtual Environment (Recommended)
**Why use a virtual environment?**
- Keeps your project dependencies isolated
- Prevents conflicts with other Python projects
- Makes it easier to manage packages

**Create and activate virtual environment:**

**On Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

**You'll know it's activated when you see `(venv)` at the start of your terminal prompt!**

### Step 3: Install Dependencies
With your virtual environment activated, run:

```bash
pip install -r requirements.txt
```

This will install:
- **PyTorch**: Deep learning framework
- **Pygame**: Game development library
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Visualization libraries

**Note:** Installation may take a few minutes, especially PyTorch.

### Step 4: Verify Installation
Test that everything works:

```bash
# Test the game
python game/flappy_bird.py

# Test the network
python agent/dqn_network.py

# Test the agent
python agent/dqn_agent.py
```

---

## 🎮 How to Use

### Step 1: Train the Agent

Train your AI agent (this will take time - be patient!):

```bash
python training/train.py
```

**What happens:**
- Agent plays 2000 games
- Starts completely random (exploring)
- Gradually learns better strategies
- Saves model every 100 episodes
- Final model saved as `models/flappy_bird_dqn_final.pth`

**Training Time:**
- Without rendering: ~30-60 minutes (depending on your computer)
- With rendering: Much slower (not recommended for training)

**Expected Progress:**
- Episodes 1-100: Random play, scores 0-5
- Episodes 100-500: Starting to learn, scores 5-20
- Episodes 500-1000: Getting better, scores 20-50
- Episodes 1000+: Expert level, scores 50-200+

### Step 2: Watch the Trained Agent Play

After training, watch your AI play:

```bash
python testing/play.py
```

This will:
- Load the trained model
- Show the game visually
- Let you watch 5 games
- Display performance statistics

### Step 3: Visualize Training Progress

See how the agent learned:

```bash
python visualization/plot_results.py
```

This creates graphs showing:
- Score improvement over time
- Learning progress (smoothed)
- Training loss
- Exploration rate decay

---

## 📁 Project Structure

```
RL/
├── game/
│   ├── __init__.py
│   └── flappy_bird.py          # Game environment
├── agent/
│   ├── __init__.py
│   ├── dqn_network.py          # Neural network (brain)
│   └── dqn_agent.py            # DQN agent (decision maker)
├── training/
│   └── train.py                # Training script
├── testing/
│   └── play.py                 # Watch AI play
├── visualization/
│   └── plot_results.py         # Plot training metrics
├── models/                     # Saved trained models (created after training)
├── logs/                       # Training metrics (created after training)
├── assets/                     # Game assets (optional)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🧠 How It Works

### The Learning Process

1. **Initialization**
   - Create game environment
   - Create neural network with random weights
   - Initialize agent with high exploration (epsilon = 1.0)

2. **Playing Games (Episodes)**
   ```
   For each episode:
       a. Reset game
       b. While not dead:
           - Observe current state
           - Choose action (random or using network)
           - Take action in game
           - Get reward and new state
           - Store experience in memory
           - Learn from random batch of memories
       c. Record score
       d. Decrease exploration rate
   ```

3. **Learning Step (The Magic!)**
   ```
   a. Sample random batch from memory (32 experiences)
   b. For each experience:
      - Predict Q-value: Q(state, action)
      - Calculate target: reward + gamma * max(Q(next_state))
      - Calculate loss: (target - Q(state, action))²
   c. Update network weights using backpropagation
   ```

### Key Concepts

**Q-Learning:**
- Q(state, action) = "How good is this action in this state?"
- Higher Q-value = better action
- Network learns to predict Q-values accurately

**Experience Replay:**
- Store past experiences: (state, action, reward, next_state, done)
- Learn from random samples (not consecutive)
- Breaks correlation between experiences

**Epsilon-Greedy:**
- **Exploration**: Try random actions (learn new things)
- **Exploitation**: Use learned knowledge (play well)
- Start with 100% exploration, end with 1% exploration

**Target Network:**
- Two networks: main (updates frequently) and target (updates slowly)
- Prevents "chasing moving target" problem
- Makes learning more stable

---

## ⚙️ Hyperparameters Explained

You can modify these in `training/train.py` to experiment:

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| **episodes** | 2000 | Number of games to play |
| **learning_rate** | 0.001 | How fast the network learns (lower = slower but more stable) |
| **gamma** | 0.99 | Discount factor (how much to value future rewards) |
| **epsilon** | 1.0 | Initial exploration rate (1.0 = 100% random) |
| **epsilon_min** | 0.01 | Final exploration rate (0.01 = 1% random) |
| **epsilon_decay** | 0.995 | How fast exploration decreases |
| **memory_size** | 10000 | How many experiences to remember |
| **batch_size** | 32 | How many experiences to learn from at once |
| **target_update** | 1000 | How often to update target network |

### Experiment Ideas:
- **Higher learning_rate** (0.01): Learns faster but might be unstable
- **Lower gamma** (0.9): Values immediate rewards more
- **Larger memory** (50000): Remembers more experiences
- **More episodes** (5000): Trains longer, might learn better

---

## 🎓 Understanding the Code

### Game State (What AI Sees)

The AI sees 5 numbers:
```python
state = [
    bird_y / screen_height,      # Bird's vertical position (0-1)
    bird_velocity / 10,           # Bird's speed (normalized)
    pipe_distance / screen_width, # Distance to next pipe (0-1)
    gap_top / screen_height,      # Top of gap (0-1)
    gap_bottom / screen_height    # Bottom of gap (0-1)
]
```

### Actions (What AI Can Do)

```python
action = 0  # Don't jump (bird falls)
action = 1  # Jump (bird goes up)
```

### Rewards (What AI Feels)

```python
+0.1   # Stay alive each frame (small positive)
+10    # Pass through a pipe (big positive!)
-10    # Die (big negative!)
```

### Neural Network Architecture

```
Input Layer (5 neurons)
    ↓
Hidden Layer 1 (128 neurons) + ReLU
    ↓
Hidden Layer 2 (128 neurons) + ReLU
    ↓
Output Layer (2 neurons) = [Q(don't jump), Q(jump)]
```

---

## 🔧 Troubleshooting

### Problem: "ModuleNotFoundError"
**Solution:** Make sure you installed dependencies:
```bash
pip install -r requirements.txt
```

### Problem: Training is too slow
**Solution:** 
- Set `render=False` in `train.py`
- Reduce number of episodes for testing
- Use GPU if available (PyTorch will detect automatically)

### Problem: Agent not learning
**Solution:**
- Make sure you're training for enough episodes (at least 500)
- Check that memory is filling up (should see loss values)
- Try adjusting learning_rate or other hyperparameters

### Problem: Game window doesn't close
**Solution:** Press ESC or close the window manually

### Problem: "CUDA out of memory"
**Solution:** 
- The code automatically uses CPU if GPU isn't available
- If you have GPU issues, the code will fall back to CPU

---

## 📊 Expected Results

After training for 2000 episodes, you should see:

- **Best Score**: 50-200+ (varies)
- **Average Score (last 100)**: 30-100+
- **Learning Curve**: Starts at 0-5, gradually increases
- **Final Epsilon**: ~0.01 (mostly using learned knowledge)

---

## 🎯 Next Steps

1. **Experiment with Hyperparameters**: Try different values and see what works best
2. **Modify Rewards**: Change reward values to see how it affects learning
3. **Add Features**: Try adding more state information (e.g., next pipe after next)
4. **Try Other Algorithms**: Implement Double DQN, Dueling DQN, or PPO
5. **Improve Visualization**: Add more graphs or real-time visualization

---

## 📝 Key Takeaways

1. **Reinforcement Learning** = Learning through trial and error with rewards
2. **DQN** = Q-Learning + Neural Networks
3. **Experience Replay** = Learning from random past experiences
4. **Exploration vs Exploitation** = Balance between trying new things and using knowledge
5. **Training Takes Time** = Be patient, the agent improves gradually

---

## 🎉 Congratulations!

You've built a complete reinforcement learning system! The AI starts knowing nothing and learns to play Flappy Bird through experience, just like a human would (but much faster with enough training).

**Questions?** Review the code comments - everything is explained in detail!

---

## 📚 Additional Resources

- **Reinforcement Learning Basics**: [OpenAI Spinning Up](https://spinningup.openai.com/)
- **DQN Paper**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- **PyTorch Tutorials**: [Official PyTorch Tutorials](https://pytorch.org/tutorials/)

---

**Happy Learning! 🚀**

