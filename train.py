"""
train.py - Training Pipeline for Pacman
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from architecture import PacmanNetwork
from data import PacmanDataset


class Pipeline(nn.Module):
    """
    Complete training pipeline.

    Args:
        path: Path to pacman_dataset.pkl
    """

    def __init__(self, path: str):
        """
        Initialize the pipeline.

        Args:
            path: Path to pacman_dataset.pkl
        """
        super().__init__()

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        self.dataset = PacmanDataset(path)

        # Initialize model
        self.model = PacmanNetwork().to(self.device)

        # Loss: CrossEntropyLoss is standard for multi-class classification
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: Adam
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3  # Standard learning rate
        )

    def train(self, batch_size: int = 128, epochs: int = 50, val_ratio: float = 0.2):
        """
        Run complete training.

        Args:
            batch_size: Batch size (default 128)
            epochs: Number of epochs (default 50)
            val_ratio: Validation ratio (0.2 = 20%)

        Returns:
            None (saves model to pacman_model.pth)
        """
        print("Beginning of the training of your network...")

        # Split train/validation
        dataset_size = len(self.dataset)
        val_size = int(val_ratio * dataset_size)
        train_size = dataset_size - val_size

        train_set, val_set = random_split(self.dataset, [train_size, val_size])

        # DataLoaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Track best model
        best_val_acc = 0.0
        best_state_dict = None

        # Training loop
        for epoch in range(1, epochs + 1):

            # ----- TRAINING PHASE -----
            self.model.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # ----- VALIDATION PHASE -----
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

            # Display
            print(f"Round {epoch}/{epochs} - accuracy: {val_acc:.2%}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = self.model.state_dict()

        # Load and save best model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        torch.save(self.model.state_dict(), "pacman_model.pth")
        print(f"Model saved! (best val_acc = {best_val_acc:.4f})")


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
