"""
pacmanagent.py - Neural Network based Pacman Agent
"""

import torch
from pacman_module.game import Agent
from data import state_to_tensor, INDEX_TO_ACTION


class PacmanAgent(Agent):
    """
    Pacman agent that uses a neural network to decide actions

    The agent does imitation learning: it predicts the action that
    the expert would have taken in each state

    Args:
        model: Trained PacmanNetwork
    """

    def __init__(self, model):
        """
        Initialize agent with a trained model

        Args:
            model: PacmanNetwork trained on expert data
        """
        super().__init__()
        self.model = model.eval()  # Evaluation mode (disables dropout)

    def get_action(self, state):
        """
        Choose the best action for the current state.

        1. Get legal actions
        2. Convert GameState to feature tensor
        3. Pass tensor through network
        4. Return the best LEGAL action

        Args:
            state: GameState object representing game state

        Returns:
            Direction (NORTH, SOUTH, EAST, WEST or STOP)
        """

        # 1. Legal actions (some directions may be blocked by walls)
        legal_actions = state.getLegalPacmanActions()

        # 2. Convert GameState to feature tensor
        # unsqueeze(0) adds batch dimension: shape becomes (1, 24)
        x = state_to_tensor(state).unsqueeze(0)

        # 3. Forward pass through the network
        with torch.no_grad():  # No gradients needed (faster)
            logits = self.model(x)[0]  # Get predictions, remove batch dimension
            probs = torch.softmax(logits, dim=0)  # Convert to probabilities [0,1]
            sorted_indices = torch.argsort(probs, descending=True).tolist()  # Sort by probability

        # 4. Return best LEGAL action
        for i in sorted_indices:
            action = INDEX_TO_ACTION[i]
            if action in legal_actions:
                return action

