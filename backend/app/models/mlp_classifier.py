"""
============================================================
Custom MLP for Face Classification -- ANN Curriculum Highlight
============================================================
Demonstrates: MLP architecture, forward propagation, 
backpropagation, ReLU activation, dropout, batch normalization.

Architecture:
  Input(512) -> FC(256) -> BN -> ReLU -> Dropout(0.3)
             -> FC(128) -> BN -> ReLU -> Dropout(0.3)
             -> FC(num_classes) -> Softmax (inference)
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

from app.config import DEVICE, FACE_EMBEDDING_DIM, MODELS_DIR


class FaceClassifierMLP(nn.Module):
    """
    Custom MLP that classifies 512-d FaceNet embeddings into student IDs.
    
    ANN Concepts:
    - Each neuron: output = activation(W · input + bias)
    - ReLU(x) = max(0, x) avoids vanishing gradient problem
    - Dropout randomly zeros neurons to prevent overfitting
    - BatchNorm normalizes layer inputs for stable training
    """

    def __init__(self, num_classes: int, input_dim: int = FACE_EMBEDDING_DIM):
        super(FaceClassifierMLP, self).__init__()

        # Layer 1: Linear(512->256) + BatchNorm + ReLU + Dropout
        # Linear performs y = W·x + b (131,072 + 256 parameters)
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Normalizes across mini-batch

        # Layer 2: Linear(256->128) + BatchNorm + ReLU + Dropout
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        # Output: Linear(128->num_classes) -- raw logits
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout: randomly zeros 30% of neurons during training
        self.dropout = nn.Dropout(p=0.3)

        self.num_classes = num_classes
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Propagation through the network.
        
        Without non-linear activations (ReLU), stacking linear layers
        would collapse to a single linear transformation. ReLU introduces
        the non-linearity needed to learn complex decision boundaries.
        
        Args:
            x: (batch_size, 512) face embeddings
        Returns:
            (batch_size, num_classes) logits
        """
        # Layer 1
        x = self.fc1(x)        # Linear transform
        x = self.bn1(x)        # Normalize
        x = F.relu(x)          # Non-linearity: max(0, x)
        x = self.dropout(x)    # Regularize

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output (no activation -- CrossEntropyLoss applies LogSoftmax)
        x = self.fc3(x)
        return x

    def predict(self, embedding: torch.Tensor) -> tuple:
        """
        Inference: returns (predicted_class_index, confidence).
        Softmax converts logits to probabilities: P(i) = exp(z_i) / Σ exp(z_j)
        """
        self.eval()
        with torch.no_grad():
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            logits = self.forward(embedding)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            return predicted.item(), confidence.item()


class MLPTrainer:
    """
    Training pipeline demonstrating:
    - CrossEntropyLoss: L = -log(P(correct_class))
    - Adam optimizer: adaptive learning rate per parameter
    - Backpropagation: loss.backward() computes ∂L/∂W via chain rule
    """

    def __init__(self, model: FaceClassifierMLP, lr: float = 0.001):
        self.model = model.to(DEVICE)
        # CrossEntropyLoss = LogSoftmax + NLL -- standard for classification
        self.criterion = nn.CrossEntropyLoss()
        # Adam: combines momentum + adaptive LR (better than plain SGD)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """One epoch: forward -> loss -> backward -> update weights."""
        self.model.train()
        total_loss = 0.0
        n = 0
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            preds = self.model(embeddings)
            loss = self.criterion(preds, labels)

            # Backpropagation -- compute gradients via chain rule:
            # ∂L/∂W = ∂L/∂output · ∂output/∂hidden · ∂hidden/∂W
            self.optimizer.zero_grad()
            loss.backward()     # Autograd computes all gradients
            self.optimizer.step()  # W_new = W_old - lr * gradient

            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def train(self, embeddings: np.ndarray, labels: np.ndarray,
              epochs: int = 100, batch_size: int = 16) -> list:
        """Full training loop. Returns loss history for plotting."""
        X = torch.FloatTensor(embeddings)
        y = torch.LongTensor(labels)
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        losses = []
        for epoch in range(epochs):
            loss = self.train_epoch(loader)
            losses.append(loss)
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")
        return losses

    def save(self, filename: str = "face_mlp.pt"):
        path = MODELS_DIR / filename
        torch.save({
            "state_dict": self.model.state_dict(),
            "num_classes": self.model.num_classes,
            "input_dim": self.model.input_dim,
        }, str(path))
        print(f"    [OK] Model saved: {path}")

    @staticmethod
    def load(filename: str = "face_mlp.pt"):
        path = MODELS_DIR / filename
        if not path.exists():
            return None
        ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)
        model = FaceClassifierMLP(ckpt["num_classes"], ckpt["input_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(DEVICE)
        model.eval()
        return model
