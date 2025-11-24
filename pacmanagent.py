"""
pacmanagent.py - Agent Pacman basé sur le réseau de neurones

============================================================================
ORIGINAL (GitHub template):
----------------------------------------------------------------------------
class PacmanAgent(Agent):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def get_action(self, state):
        x = state_to_tensor(state).unsqueeze(0)
        # Your code here
        return # ...
============================================================================
"""

import torch
from pacman_module.game import Agent
from data import state_to_tensor, INDEX_TO_ACTION


class PacmanAgent(Agent):
    """
    Agent Pacman qui utilise un réseau de neurones pour décider.

    L'agent fait de l'imitation learning: il prédit l'action que
    l'expert aurait prise dans chaque état.

    Arguments:
        model: PacmanNetwork entraîné

    Attributes:
        model: Le réseau en mode évaluation
    """

    def __init__(self, model):
        """
        Initialise l'agent avec un modèle entraîné.

        Arguments:
            model: PacmanNetwork entraîné sur les données d'expert
        """
        super().__init__()
        self.model = model.eval()  # Mode évaluation (désactive dropout)

    def get_action(self, state):
        """
        Choisit la meilleure action pour l'état actuel.

        ---------- NOTRE IMPLEMENTATION ----------

        1. Récupère les actions légales
        2. Convertit le GameState en tensor de features
        3. Passe le tensor dans le réseau
        4. Retourne la meilleure action LEGALE

        Arguments:
            state: Objet GameState représentant l'état du jeu

        Returns:
            Direction (NORTH, SOUTH, EAST, WEST ou STOP)
        """
        # ---------- NOTRE CODE COMMENCE ICI ----------

        # 1. Actions légales (certaines directions peuvent être bloquées par des murs)
        legal_actions = state.getLegalPacmanActions()

        # 2. Convertir GameState en tensor de features
        # unsqueeze(0) ajoute une dimension batch: shape devient (1, 24)
        x = state_to_tensor(state).unsqueeze(0)

        # 3. Déplacer le tensor sur le même device que le modèle
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = "cpu"
        x = x.to(device)

        # 4. Forward pass
        with torch.no_grad():  # Pas de gradients pour l'inférence
            logits = self.model(x)[0]  # Enlever la dimension batch
            probs = torch.softmax(logits, dim=0)  # Convertir en probabilités
            sorted_indices = torch.argsort(probs, descending=True).tolist()

        # 5. Retourner la meilleure action LEGALE
        for idx in sorted_indices:
            action = INDEX_TO_ACTION[idx]
            if action in legal_actions:
                return action

        # Fallback (ne devrait jamais arriver)
        return legal_actions[0]

        # ---------- FIN DE NOTRE CODE ----------
