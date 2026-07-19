# 🚀 Quick Start Guide

Follow these steps to get started quickly!

## Step 1: Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Test the Game (Optional)

Make sure the game works:

```bash
python game/flappy_bird.py
```

- Click the window and press **SPACE** to make the bird jump
- Close the window when done

## Step 4: Train the AI

**This is the main step!** Train your AI agent:

```bash
python training/train.py
```

**What to expect:**
- ⏱️ Takes 30-60 minutes (depending on your computer)
- 📊 You'll see progress every 10 episodes
- 💾 Model saves every 100 episodes
- 🎯 Agent starts random, gradually learns

**Be patient!** The agent needs time to learn.

## Step 5: Watch Your AI Play!

After training completes:

```bash
python testing/play.py
```

Watch your trained AI play Flappy Bird! 🎮

## Step 6: See the Learning Progress

Visualize how the agent learned:

```bash
python visualization/plot_results.py
```

This shows graphs of:
- Score improvement
- Learning progress
- Training metrics

---

## 🎯 That's It!

You've successfully:
1. ✅ Created a Flappy Bird game
2. ✅ Built a DQN neural network
3. ✅ Trained an AI agent
4. ✅ Watched it play
5. ✅ Visualized the learning

---

## 💡 Tips

- **First time?** Start with fewer episodes (modify `episodes=500` in `train.py`)
- **Want to watch training?** Set `render=True` in `train.py` (much slower!)
- **Experiment!** Try changing hyperparameters in `train.py`
- **Read the code!** Everything is heavily commented to help you learn

---

## 📚 Need More Help?

Check the main `README.md` for detailed explanations of everything!

---

**Happy Learning! 🎓**

