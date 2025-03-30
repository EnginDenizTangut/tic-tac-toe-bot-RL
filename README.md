# Tic-Tac-Toe with Q-Learning AI

## Description
This Python program implements a Tic-Tac-Toe game where you can play against an AI opponent trained using Q-learning reinforcement learning. The AI learns optimal strategies through thousands of simulated games.

## Features
- Classic 3×3 Tic-Tac-Toe implementation
- Q-learning based AI opponent
- Training mode with configurable number of episodes
- Interactive gameplay against the trained AI
- Score tracking during training (wins/losses/draws)

## Requirements
- Python 3.x
- NumPy (`pip install numpy`)

## How It Works

### Core Components:
1. **TicTacToe Class**:
   - Manages game state and logic
   - Handles player turns
   - Checks for winners/draws

2. **QLearningRobot Class**:
   - Implements Q-learning algorithm
   - Maintains Q-table for state-action values
   - Uses epsilon-greedy strategy for exploration/exploitation

3. **Training Process**:
   - The AI plays against a random opponent for many episodes
   - Updates Q-values based on game outcomes
   - Learns optimal moves through reinforcement

## Usage

### Training the AI:
```python
robot = train_robot(episodes=100000)  # Train with 100,000 games
```

### Playing Against the AI:
```python
play_against_robot(trained_robot)
```

## Key Parameters
- `learning_rate = 0.1` (How quickly the AI adapts to new information)
- `discount_factor = 0.9` (How much the AI values future rewards)
- `epsilon = 0.2` (Exploration rate - chance of random moves during training)

## Training Notes
- The default training uses 1,000,000 episodes for thorough learning
- Training results show win percentages for both players
- The AI ('O') learns to never lose (either wins or draws) with sufficient training

## Gameplay Instructions
1. The board positions are numbered 0-8 (left to right, top to bottom)
2. You play as 'X', the AI is 'O'
3. Enter your move by typing the position number (0-8) when prompted

## Example Output
```
Robot eğitiliyor...
Eğitim sonuçları: {'X': 1243, 'O': 7231, 'Draw': 1526}

Oyuna hoş geldiniz! Sen 'X', robot 'O' oynuyor.
X |   |  
---------
  |   |  
---------
  |   |  
Hamlenizi girin (0-8):
```

## Customization
- Adjust training parameters in the `QLearningRobot` class
- Change number of training episodes in `train_robot()` call
- Modify reward values in the training loop for different learning behaviors
