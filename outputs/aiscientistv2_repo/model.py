"""
model.py

This module defines two network architectures:
  1. LSTMModel: An LSTM-based model with compositional regularization for synthetic arithmetic experiments.
     - Architecture: An Embedding layer, a single LSTM layer, and a Linear layer for regression.
     - Additionally, computes a compositional regularization loss based on the squared differences
       between successive input embeddings. This loss is stored as an attribute (self.compositional_loss)
       for later use in loss aggregation.
       
  2. ResNet18Model: A ResNet-18 based model for image experiments (label noise calibration and pest detection).
     - Architecture: A manually implemented ResNet-18 using BasicBlock modules.
     - The network adapts its final fully connected layer to output logits for a specified number of classes.

Both classes inherit from torch.nn.Module and expose a forward(x: Tensor) -> Tensor interface.
Configuration values are read from a parameters dictionary (with default values provided).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

# =========================
# LSTMModel for Synthetic Arithmetic
# =========================
class LSTMModel(nn.Module):
    """
    LSTM-based model with compositional regularization.

    Attributes:
        embedding (nn.Embedding): Embedding layer converting token indices to vectors.
        lstm (nn.LSTM): Single-layer LSTM processing the sequence of embeddings.
        fc (nn.Linear): Fully connected layer mapping the LSTM output to a regression output.
        compositional_loss (torch.Tensor): Stores the computed compositional loss based on successive embedding differences.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the LSTMModel architecture.

        Args:
            params (Optional[Dict[str, Any]]): Dictionary of hyperparameters. Expected keys include:
                - 'vocab_size': (int) Number of tokens in the vocabulary (default: 50).
                - 'embedding_dim': (int) Dimension of the embedding vectors (default: 16).
                - 'lstm_hidden_size': (int) Hidden size for the LSTM (default: 32).
                - 'lstm_num_layers': (int) Number of LSTM layers (default: 1).
                - 'output_size': (int) Dimension of the final output (default: 1).
        """
        super(LSTMModel, self).__init__()
        if params is None:
            params = {}

        self.vocab_size: int = int(params.get("vocab_size", 50))
        self.embedding_dim: int = int(params.get("embedding_dim", 16))
        self.lstm_hidden_size: int = int(params.get("lstm_hidden_size", 32))
        self.lstm_num_layers: int = int(params.get("lstm_num_layers", 1))
        self.output_size: int = int(params.get("output_size", 1))

        # Embedding layer: Converts token indices to embeddings of dimension embedding_dim.
        self.embedding: nn.Embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                                    embedding_dim=self.embedding_dim)

        # LSTM layer: Processes sequence of embeddings.
        self.lstm: nn.LSTM = nn.LSTM(input_size=self.embedding_dim,
                                     hidden_size=self.lstm_hidden_size,
                                     num_layers=self.lstm_num_layers,
                                     batch_first=True)

        # Fully connected (linear) layer: Maps LSTM output at the last time step to output.
        self.fc: nn.Linear = nn.Linear(in_features=self.lstm_hidden_size,
                                       out_features=self.output_size)

        # Initialize compositional_loss as a tensor (will be computed in forward).
        self.compositional_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTMModel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length) containing token indices.

        Returns:
            torch.Tensor: Prediction tensor of shape (batch_size, output_size).
        """
        # Compute embeddings: shape (batch_size, sequence_length, embedding_dim)
        embeddings: torch.Tensor = self.embedding(x)

        # Compute compositional regularization loss:
        # For each sequence, compute the mean squared difference between successive embedding vectors.
        if embeddings.size(1) > 1:
            diff: torch.Tensor = embeddings[:, 1:, :] - embeddings[:, :-1, :]
            self.compositional_loss = torch.mean(diff ** 2)
        else:
            # If sequence length is 1, no difference can be computed.
            self.compositional_loss = torch.tensor(0.0, device=embeddings.device)

        # Pass embeddings through LSTM: lstm_out shape -> (batch_size, sequence_length, lstm_hidden_size)
        lstm_out, _ = self.lstm(embeddings)
        # Use the output at the last time step as the representation for prediction.
        last_output: torch.Tensor = lstm_out[:, -1, :]
        prediction: torch.Tensor = self.fc(last_output)
        return prediction

# =========================
# ResNet-18 Implementation for Image Experiments
# =========================

def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """
    3x3 convolution with padding.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the convolution.

    Returns:
        nn.Conv2d: 3x3 convolution layer.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    """
    Basic block for ResNet-18.

    Attributes:
        conv1 (nn.Conv2d): First convolution.
        bn1 (nn.BatchNorm2d): Batch normalization for conv1.
        conv2 (nn.Conv2d): Second convolution.
        bn2 (nn.BatchNorm2d): Batch normalization for conv2.
        downsample (Optional[nn.Module]): Downsampling layer, if needed.
        relu (nn.ReLU): ReLU activation function.
    """
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None) -> None:
        """
        Initialize a BasicBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolution.
            downsample (Optional[nn.Module]): Optional downsampling module.
        """
        super(BasicBlock, self).__init__()
        self.conv1: nn.Conv2d = conv3x3(in_channels, out_channels, stride)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.conv2: nn.Conv2d = conv3x3(out_channels, out_channels)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.downsample: Optional[nn.Module] = downsample
        self.stride: int = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the BasicBlock.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after adding the residual connection.
        """
        identity: torch.Tensor = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet18Model(nn.Module):
    """
    ResNet-18 based model for image experiments.

    Attributes:
        conv1 (nn.Conv2d): Initial convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization for conv1.
        relu (nn.ReLU): ReLU activation.
        maxpool (nn.MaxPool2d): Max pooling layer.
        layer1, layer2, layer3, layer4 (nn.Sequential): Stacked BasicBlock layers.
        avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        fc (nn.Linear): Final fully connected layer mapping to output classes.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the ResNet18Model.

        Args:
            params (Optional[Dict[str, Any]]): Dictionary of model parameters. Expected keys include:
                - 'num_classes': (int) Number of target classes (default: 10).
        """
        super(ResNet18Model, self).__init__()
        if params is None:
            params = {}
        self.num_classes: int = int(params.get("num_classes", 10))
        self.in_channels: int = 64

        # Initial convolutional block
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=3, out_channels=64,
                                           kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(64)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.maxpool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers: following the ResNet-18 configuration [2, 2, 2, 2]
        self.layer1: nn.Sequential = self._make_layer(BasicBlock, out_channels=64, blocks=2, stride=1)
        self.layer2: nn.Sequential = self._make_layer(BasicBlock, out_channels=128, blocks=2, stride=2)
        self.layer3: nn.Sequential = self._make_layer(BasicBlock, out_channels=256, blocks=2, stride=2)
        self.layer4: nn.Sequential = self._make_layer(BasicBlock, out_channels=512, blocks=2, stride=2)

        # Average pooling and fully connected classifier.
        self.avgpool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc: nn.Linear = nn.Linear(512 * BasicBlock.expansion, self.num_classes)

    def _make_layer(self, block: nn.Module, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """
        Create a sequential layer consisting of several BasicBlocks.

        Args:
            block (nn.Module): The block type to use (BasicBlock).
            out_channels (int): Number of output channels for the layer.
            blocks (int): Number of blocks in the layer.
            stride (int): Stride for the first block.

        Returns:
            nn.Sequential: The stacked blocks forming the layer.
        """
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ResNet18Model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits: torch.Tensor = self.fc(x)
        return logits

# End of model.py
