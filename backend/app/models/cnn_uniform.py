"""
============================================================
Custom CNN for Uniform Classification -- ANN Curriculum Highlight
============================================================
Demonstrates: Convolutional layers, pooling, feature maps,
spatial feature extraction, and multi-label classification.

Architecture:
  Conv2d(3,32,3) -> ReLU -> MaxPool(2)
  Conv2d(32,64,3) -> ReLU -> MaxPool(2)
  Conv2d(64,128,3) -> ReLU -> MaxPool(2)
  Flatten -> FC(128*28*28, 256) -> ReLU -> Dropout(0.5)
          -> FC(256, 4) [shirt, pant, tucked, not_uniform]

ANN Concepts:
  - Convolution: sliding kernel extracts local spatial features
    (edges, textures, color patterns) -- weight sharing across
    spatial locations makes CNNs efficient for image tasks.
  - Pooling: reduces spatial dimensions and provides translation
    invariance (small shifts don't change the output).
  - Feature hierarchy: shallow layers detect edges/colors,
    deeper layers detect complex patterns (shirt collars, etc.)
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from app.config import DEVICE, MODELS_DIR, UNIFORM_IMAGE_SIZE


class UniformClassifierCNN(nn.Module):
    """
    CNN that classifies body-region crops into uniform components.
    
    Input: (batch, 3, 224, 224) RGB image crop
    Output: (batch, 4) -- logits for [shirt, pant, tucked, not_uniform]
    
    Convolution explained:
    - A 3×3 kernel slides across the image
    - At each position, it computes: Σ(kernel * patch) + bias
    - This detects local patterns (edges, textures, colors)
    - Multiple kernels = multiple feature maps = richer representation
    """

    def __init__(self, num_classes: int = 4):
        super(UniformClassifierCNN, self).__init__()

        # ── Convolutional Feature Extractor ──────────────
        # Conv2d(in_channels, out_channels, kernel_size, padding)
        
        # Block 1: Detect low-level features (edges, color gradients)
        # 3 input channels (RGB) -> 32 feature maps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # MaxPool2d(2): reduces 224×224 -> 112×112 (halves spatial dims)
        # Takes maximum value in each 2×2 window -- provides
        # translation invariance and reduces computation
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: Detect mid-level features (textures, patterns)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112×112 -> 56×56

        # Block 3: Detect high-level features (shirt collars, waistline)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56×56 -> 28×28

        # ── Classifier Head (Fully Connected) ────────────
        # Flatten 128 feature maps of 28×28 = 100,352 features
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.
        
        Convolution operation at each layer:
          output[i,j] = Σ_m Σ_n kernel[m,n] * input[i+m, j+n] + bias
        
        This is NOT matrix multiplication -- it's a sliding dot product
        that preserves spatial relationships in the image.
        """
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # -> (batch, 32, 112, 112)

        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # -> (batch, 64, 56, 56)

        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # -> (batch, 128, 28, 28)

        # Flatten spatial dimensions for the FC layers
        x = x.view(x.size(0), -1)  # -> (batch, 100352)

        # Fully connected classifier
        x = F.relu(self.fc1(x))     # -> (batch, 256)
        x = self.dropout(x)
        x = self.fc2(x)            # -> (batch, num_classes)

        return x

    def predict(self, image_tensor: torch.Tensor) -> dict:
        """
        Inference: returns dict of component confidences.
        Uses sigmoid (not softmax) because labels are multi-label:
        a single image can be both "shirt" AND "tucked".
        
        Sigmoid: σ(x) = 1 / (1 + e^(-x)) -- squashes each logit to [0, 1]
        """
        self.eval()
        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            logits = self.forward(image_tensor)
            probs = torch.sigmoid(logits)  # Multi-label probabilities
            probs = probs.squeeze(0).cpu().numpy()

        labels = ["shirt", "pant", "tucked", "not_uniform"]
        return {label: float(prob) for label, prob in zip(labels, probs)}


class CNNTrainer:
    """
    Training pipeline for the Uniform CNN.
    Uses BCEWithLogitsLoss for multi-label classification:
      L = -[y·log(σ(x)) + (1-y)·log(1-σ(x))]
    This is Binary Cross-Entropy applied per label independently.
    """

    def __init__(self, model: UniformClassifierCNN, lr: float = 0.0001):
        self.model = model.to(DEVICE)
        # BCEWithLogitsLoss: combines sigmoid + BCE for numerical stability
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = self.model(images)
            loss = self.criterion(preds, labels.float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def train(self, images: np.ndarray, labels: np.ndarray,
              epochs: int = 50, batch_size: int = 8) -> list:
        X = torch.FloatTensor(images)
        y = torch.FloatTensor(labels)
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        losses = []
        for epoch in range(epochs):
            loss = self.train_epoch(loader)
            losses.append(loss)
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")
        return losses

    def save(self, filename: str = "uniform_cnn.pt"):
        path = MODELS_DIR / filename
        torch.save({
            "state_dict": self.model.state_dict(),
            "num_classes": self.model.num_classes,
        }, str(path))

    @staticmethod
    def load(filename: str = "uniform_cnn.pt"):
        path = MODELS_DIR / filename
        if not path.exists():
            return None
        ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)
        model = UniformClassifierCNN(ckpt["num_classes"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(DEVICE)
        model.eval()
        return model
