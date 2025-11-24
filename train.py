"""
train.py - Pipeline d'entraînement pour Pacman

============================================================================
ORIGINAL (GitHub template):
----------------------------------------------------------------------------
class Pipeline(nn.Module):
    def __init__(self, path):
        self.dataset = PacmanDataset(path)
        self.model = PacmanNetwork()
        self.criterion = # Your code here
        self.optimizer = # Your code here

    def train(self):
        print("Beginning of the training of your network...")
        # Your code here
        torch.save(self.model.state_dict(), "pacman_model.pth")
============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from architecture import PacmanNetwork
from data import PacmanDataset


class Pipeline(nn.Module):
    """
    Pipeline d'entraînement complète.

    Arguments:
        path: Chemin vers pacman_dataset.pkl

    Attributes:
        device: CPU ou CUDA
        dataset: PacmanDataset chargé
        model: PacmanNetwork
        criterion: Fonction de loss
        optimizer: Optimiseur
        scheduler: Learning rate scheduler
    """

    def __init__(self, path: str):
        """
        Initialise la pipeline.

        Arguments:
            path: Chemin vers pacman_dataset.pkl
        """
        super().__init__()

        # ---------- NOTRE CODE COMMENCE ICI ----------

        # Utiliser GPU si disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Charger le dataset
        self.dataset = PacmanDataset(path)

        # Initialiser le modèle
        self.model = PacmanNetwork().to(self.device)

        # Loss: CrossEntropyLoss est standard pour la classification multi-classe
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: Adam avec weight_decay pour régularisation L2
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,           # Learning rate standard
            weight_decay=1e-4  # Régularisation L2
        )

        # Scheduler: réduit lr quand val_loss stagne
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',    # Minimiser val_loss
            factor=0.5,    # Diviser lr par 2
            patience=5     # Attendre 5 epochs avant de réduire
        )

        # ---------- FIN DE NOTRE CODE ----------

    def train(self, batch_size: int = 128, epochs: int = 50, val_ratio: float = 0.2):
        """
        Lance l'entraînement complet.

        Arguments:
            batch_size: Taille des batches (128 par défaut)
            epochs: Nombre d'epochs (50 par défaut)
            val_ratio: Ratio de validation (0.2 = 20%)

        Returns:
            None (sauvegarde le modèle dans pacman_model.pth)
        """
        print("Beginning of the training of your network...")

        # ---------- NOTRE CODE COMMENCE ICI ----------

        # Split train/validation
        dataset_size = len(self.dataset)
        val_size = int(val_ratio * dataset_size)
        train_size = dataset_size - val_size

        train_set, val_set = random_split(self.dataset, [train_size, val_size])

        # DataLoaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Tracking du meilleur modèle
        best_val_acc = 0.0
        best_state_dict = None

        # Boucle d'entraînement
        for epoch in range(1, epochs + 1):

            # ----- PHASE TRAINING -----
            self.model.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # ----- PHASE VALIDATION -----
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device, dtype=torch.long)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_loss / val_total if val_total > 0 else 0.0
            val_acc = val_correct / val_total if val_total > 0 else 0.0

            # Affichage
            print(f"Round {epoch}/{epochs} - accuracy: {val_acc:.2%}")

            # Update scheduler
            self.scheduler.step(val_loss)

            # Sauvegarder le meilleur modèle
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = self.model.state_dict()

        # ---------- FIN DE NOTRE CODE ----------

        # Charger et sauvegarder le meilleur modèle
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        torch.save(self.model.state_dict(), "pacman_model.pth")
        print(f"Model saved! (best val_acc = {best_val_acc:.4f})")


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
