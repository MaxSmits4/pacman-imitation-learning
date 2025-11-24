"""
architecture.py - Neural Network for Pacman Imitation Learning
"""

import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    ============================================================================
    ORIGINAL (GitHub template):
    ----------------------------------------------------------------------------
    class PacmanNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # Your code here

        def forward(self, x):
            # Your code here
            return # ...
    ============================================================================

    NOTRE IMPLEMENTATION:
    ---------------------
    MLP avec BatchNorm pour la stabilité de l'entraînement.
    - Input: 24 features (extraites par state_to_tensor dans data.py)
    - Output: 5 logits (une par action possible)
    - Architecture: 24 → 128 → 128 → 64 → 5

    Pourquoi ces choix:
    - BatchNorm: stabilise les gradients, accélère la convergence
    - ReLU: activation simple et efficace, pas de vanishing gradient
    - Dropout 0.1: légère régularisation (BatchNorm régularise déjà)
    - Pas d'activation finale: CrossEntropyLoss attend des logits bruts
    """

    def __init__(self):
        """
        Initialise les couches du réseau.

        Arguments: Aucun (architecture fixe)
        Returns: None
        """
        super().__init__()

        # ---------- NOTRE CODE COMMENCE ICI ----------

        input_dim = 24   # Doit correspondre à state_to_tensor()
        output_dim = 5   # NORTH, SOUTH, EAST, WEST, STOP

        # Architecture avec BatchNorm pour la stabilité
        self.net = nn.Sequential(
            # Couche 1: 24 → 128
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            # Couche 2: 128 → 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            # Couche 3: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            # Couche de sortie: 64 → 5 (pas d'activation, CE veut des logits)
            nn.Linear(64, output_dim)
        )

        # ---------- FIN DE NOTRE CODE ----------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant dans le réseau.

        Arguments:
            x: Tensor de shape (batch_size, 24) - features normalisées

        Returns:
            Tensor de shape (batch_size, 5) - logits pour chaque action
        """
        # ---------- NOTRE CODE COMMENCE ICI ----------
        return self.net(x)
        # ---------- FIN DE NOTRE CODE ----------
