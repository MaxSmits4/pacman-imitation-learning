# Pacman - Imitation Learning

Project for **INFO8006 - Introduction to Artificial Intelligence** (ULiège).

## Description

Train a neural network to imitate an expert Pacman player. The model learns from (game state, expert action) pairs and predicts which action to take in each situation.

## Installation

```bash
pip install torch pandas
```

## Usage

**1. Train the model:**
```bash
python train.py
```
This generates `pacman_model.pth`.

**2. Run the game:**
```bash
python run.py
```

## Structure

- `architecture.py` - Neural network (MLP)
- `data.py` - Feature extraction and dataset
- `train.py` - Training pipeline
- `pacmanagent.py` - Agent using the trained model
- `run.py` - Run a game with the trained agent
