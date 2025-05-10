"""
trainer.py

This module implements the Trainer class, which is responsible for running the training loop 
for a given model over a provided training dataset. It computes the main loss as well as any
additional regularization loss (for experiments such as compositional regularization), performs 
backpropagation, updates model parameters using the provided optimizer (and scheduler), saves 
checkpoints, and records training metrics.

The Trainer class exposes two primary methods:
  - train() -> dict: Runs the training loop for a specified number of epochs and returns a dictionary 
      of aggregated training metrics.
  - validate() -> dict: Evaluates the model on a validation dataset and returns metrics.
      
The Trainer uses configuration settings from config.yaml (passed as a dictionary) and relies on 
an experiment_type parameter to distinguish between different experimental settings:
  * "synthetic_arithmetic" for compositional regularization experiments (regression with MSE loss and 
    an additional compositional loss computed on successive embedding differences),
  * "label_noise" for experiments using cross-entropy loss (e.g. classification on CIFAR-10 with label noise),
  * "pest_detection" for image classification tasks (using a ResNet-18 model).

All default values are explicitly set and strong type hints are provided.

Author: [Your Name]
Date: [Date]
"""

import os
import logging
import random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm  # progress bar for epochs and batches

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Trainer:
    """
    Trainer class responsible for model training and validation.

    Attributes:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (Optional[Any]): Learning rate scheduler (if provided).
        train_data (Any): The training dataset (list or DataLoader).
        config (Dict[str, Any]): The configuration dictionary.
        experiment_type (str): Type of experiment. Expected values: 
                                "synthetic_arithmetic", "label_noise", "pest_detection".
        train_loader (DataLoader): DataLoader for the training dataset.
        device (torch.device): Device to run the training on.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        reg_weight (float): Regularization weight (used only in synthetic_arithmetic experiment).
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 scheduler: Optional[Any], data: Any, config: Dict[str, Any],
                 experiment_type: str) -> None:
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): Instance of the model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer instance.
            scheduler (Optional[Any]): Learning rate scheduler instance (can be None).
            data (Any): Training dataset (list or DataLoader).
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
            experiment_type (str): Specifies the experiment type:
                                   "synthetic_arithmetic", "label_noise", or "pest_detection".
        """
        self.config: Dict[str, Any] = config
        self.experiment_type: str = experiment_type.lower()
        self.model: nn.Module = model
        self.optimizer: optim.Optimizer = optimizer
        self.scheduler: Optional[Any] = scheduler

        # Set device and move model to device.
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Determine the specific training config based on experiment type.
        if self.experiment_type == "synthetic_arithmetic":
            self.training_config: Dict[str, Any] = self.config.get("training", {}).get("compositional_reg", {})
        elif self.experiment_type == "label_noise":
            self.training_config = self.config.get("training", {}).get("label_noise", {})
        elif self.experiment_type == "pest_detection":
            self.training_config = self.config.get("training", {}).get("pest_detection", {})
        else:
            logging.warning("Unknown experiment type provided; defaulting to pest_detection settings.")
            self.training_config = self.config.get("training", {}).get("pest_detection", {})

        # Set hyperparameters with defaults.
        self.epochs: int = int(self.training_config.get("epochs", 30))
        self.batch_size: int = int(self.training_config.get("batch_size", 32))
        self.reg_weight: float = float(self.training_config.get("regularization_weight", 0.1))
        
        # Set random seeds to ensure reproducibility.
        seed: int = int(self.config.get("general", {}).get("random_seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Prepare DataLoader if data is not already a DataLoader.
        if isinstance(data, DataLoader):
            self.train_loader: DataLoader = data
        else:
            self.train_loader: DataLoader = DataLoader(
                dataset=data,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self._collate_fn
            )

        logging.info(f"Trainer initialized for experiment_type: {self.experiment_type}, "
                     f"epochs: {self.epochs}, batch_size: {self.batch_size}, reg_weight: {self.reg_weight}")

        # Define loss functions depending on experiment type.
        if self.experiment_type == "synthetic_arithmetic":
            self.main_loss_fn = nn.MSELoss()
            # For regression accuracy, we use a tolerance value (default 1.0)
            self.regression_tolerance: float = 1.0
            self._calculate_accuracy = self._calc_regression_accuracy
        else:
            self.main_loss_fn = nn.CrossEntropyLoss()
            self._calculate_accuracy = self._calc_classification_accuracy

    def _collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to process raw data samples into batched tensors.
        
        For 'synthetic_arithmetic' experiment, converts string expressions into tensor of token indices.
        For 'label_noise' and 'pest_detection' experiments, converts images (numpy arrays) and labels into tensors.

        Args:
            batch (List[Any]): List of data samples (dictionaries).

        Returns:
            Dict[str, torch.Tensor]: Dictionary with keys "input" and "target".
        """
        if self.experiment_type == "synthetic_arithmetic":
            inputs: List[torch.Tensor] = []
            targets: List[torch.Tensor] = []
            # Define a mapping for characters to token indices.
            mapping: Dict[str, int] = {str(i): i for i in range(10)}
            mapping.update({"+": 10, "*": 11})
            for sample in batch:
                expr_str: str = sample.get("input", "")
                # Convert each character in the expression to its corresponding token.
                token_list: List[int] = [mapping.get(ch, 0) for ch in expr_str]
                inputs.append(torch.tensor(token_list, dtype=torch.long))
                targets.append(torch.tensor(sample.get("target", 0), dtype=torch.float32))
            inputs_tensor: torch.Tensor = torch.stack(inputs)  # Shape: (batch_size, seq_length)
            targets_tensor: torch.Tensor = torch.stack(targets)  # Shape: (batch_size,)
            return {"input": inputs_tensor, "target": targets_tensor}
        elif self.experiment_type in {"label_noise", "pest_detection"}:
            # Check if data sample contains images.
            if "image" in batch[0]:
                images: List[torch.Tensor] = []
                labels: List[torch.Tensor] = []
                for sample in batch:
                    img: Any = sample.get("image")
                    if isinstance(img, np.ndarray):
                        img_tensor = torch.from_numpy(img).float()
                        # If image is in H x W x C format, permute to C x H x W.
                        if img_tensor.ndim == 3 and img_tensor.shape[2] in [1, 3]:
                            img_tensor = img_tensor.permute(2, 0, 1)
                        images.append(img_tensor)
                    else:
                        images.append(img)
                    labels.append(torch.tensor(sample.get("label", 0), dtype=torch.long))
                images_tensor: torch.Tensor = torch.stack(images)
                labels_tensor: torch.Tensor = torch.stack(labels)
                return {"input": images_tensor, "target": labels_tensor}
            else:
                # Fallback: assume dictionaries with "input" and "target" that are numerical.
                inputs = [torch.tensor(sample["input"]) for sample in batch]
                targets = [torch.tensor(sample["target"]) for sample in batch]
                return {"input": torch.stack(inputs), "target": torch.stack(targets)}
        else:
            raise ValueError("Unsupported experiment type in collate function.")

    def _calc_regression_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor,
                                    tolerance: Optional[float] = None) -> float:
        """
        Calculate regression accuracy as the fraction of predictions within a tolerance of target values.

        Args:
            predictions (torch.Tensor): Model predictions (shape: [batch_size, 1] or [batch_size]).
            targets (torch.Tensor): Ground truth values (shape: [batch_size]).
            tolerance (Optional[float]): Allowed absolute difference. Default uses self.regression_tolerance.

        Returns:
            float: Accuracy value between 0 and 1.
        """
        if tolerance is None:
            tolerance = self.regression_tolerance
        # Ensure predictions are squeezed to shape (batch_size,)
        preds = predictions.squeeze()
        # Compute absolute differences and check within tolerance.
        correct = (torch.abs(preds - targets) <= tolerance).float().mean().item()
        return correct

    def _calc_classification_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate classification accuracy by comparing predicted labels to ground-truth labels.

        Args:
            predictions (torch.Tensor): Logit outputs (shape: [batch_size, num_classes]).
            targets (torch.Tensor): Ground truth labels (shape: [batch_size]).

        Returns:
            float: Accuracy as a fraction between 0 and 1.
        """
        pred_labels = torch.argmax(predictions, dim=1)
        correct = (pred_labels == targets).float().mean().item()
        return correct

    def _save_checkpoint(self, epoch: int) -> str:
        """
        Save a checkpoint of the current model state.

        Args:
            epoch (int): Current epoch number.

        Returns:
            str: The checkpoint file path.
        """
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        checkpoint_path: str = f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved at {checkpoint_path}")
        return checkpoint_path

    def train(self) -> Dict[str, Any]:
        """
        Run the training loop over the training dataset.

        Returns:
            Dict[str, Any]: Aggregated training metrics, including epoch losses and accuracies.
        """
        self.model.train()
        epoch_losses: List[float] = []
        epoch_main_losses: List[float] = []
        epoch_comp_losses: List[float] = []
        epoch_accuracies: List[float] = []

        for epoch in range(1, self.epochs + 1):
            running_loss: float = 0.0
            running_main_loss: float = 0.0
            running_comp_loss: float = 0.0
            running_accuracy: float = 0.0
            batch_count: int = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}", leave=False)
            for batch in progress_bar:
                try:
                    inputs: torch.Tensor = batch["input"].to(self.device)
                    targets: torch.Tensor = batch["target"].to(self.device)

                    self.optimizer.zero_grad()
                    # Forward pass
                    outputs: torch.Tensor = self.model(inputs)
                    
                    # Compute main loss and, if synthetic_arithmetic experiment, include compositional loss.
                    if self.experiment_type == "synthetic_arithmetic":
                        # In synthetic arithmetic, the main loss is MSE between prediction and target.
                        # Ensure outputs is of shape (batch, 1) and targets is (batch,)
                        main_loss: torch.Tensor = self.main_loss_fn(outputs.squeeze(), targets)
                        comp_loss: torch.Tensor = self.model.compositional_loss
                        total_loss: torch.Tensor = main_loss + self.reg_weight * comp_loss
                    else:
                        # For classification experiments use CrossEntropyLoss.
                        main_loss = self.main_loss_fn(outputs, targets)
                        total_loss = main_loss
                        comp_loss = torch.tensor(0.0)  # No compositional loss

                    # Backward pass and optimizer step.
                    total_loss.backward()
                    self.optimizer.step()

                    # Calculate batch accuracy.
                    batch_accuracy: float = self._calculate_accuracy(outputs, targets)

                    running_loss += total_loss.item()
                    running_main_loss += main_loss.item()
                    running_comp_loss += comp_loss.item()
                    running_accuracy += batch_accuracy
                    batch_count += 1

                    progress_bar.set_postfix({
                        "Loss": total_loss.item(),
                        "MainLoss": main_loss.item(),
                        "CompLoss": comp_loss.item(),
                        "Acc": batch_accuracy
                    })
                except Exception as e:
                    logging.error(f"Error during training at epoch {epoch}, batch {batch_count}: {e}")
                    return {"error": str(e)}

            # Compute epoch averages.
            avg_total_loss: float = running_loss / batch_count if batch_count > 0 else 0.0
            avg_main_loss: float = running_main_loss / batch_count if batch_count > 0 else 0.0
            avg_comp_loss: float = running_comp_loss / batch_count if batch_count > 0 else 0.0
            avg_accuracy: float = running_accuracy / batch_count if batch_count > 0 else 0.0

            epoch_losses.append(avg_total_loss)
            epoch_main_losses.append(avg_main_loss)
            epoch_comp_losses.append(avg_comp_loss)
            epoch_accuracies.append(avg_accuracy)

            # Step the scheduler if provided.
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint after each epoch.
            self._save_checkpoint(epoch)

            logging.info(f"Epoch {epoch}/{self.epochs} - Avg Total Loss: {avg_total_loss:.4f}, "
                         f"Avg Main Loss: {avg_main_loss:.4f}, Avg Comp Loss: {avg_comp_loss:.4f}, "
                         f"Avg Accuracy: {avg_accuracy:.4f}")

        # Return aggregated training metrics.
        return {
            "epoch_losses": epoch_losses,
            "epoch_main_loss": epoch_main_losses,
            "epoch_comp_loss": epoch_comp_losses,
            "epoch_accuracy": epoch_accuracies
        }

    def validate(self, val_data: Any) -> Dict[str, Any]:
        """
        Evaluate the model on a validation dataset and compute loss and accuracy metrics.

        Args:
            val_data (Any): Validation dataset (list or DataLoader).

        Returns:
            Dict[str, Any]: Dictionary with validation loss and accuracy metrics.
        """
        # If val_data is not a DataLoader, wrap it.
        if not isinstance(val_data, DataLoader):
            val_loader = DataLoader(
                dataset=val_data,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self._collate_fn
            )
        else:
            val_loader = val_data

        self.model.eval()
        total_loss: float = 0.0
        total_accuracy: float = 0.0
        batch_count: int = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                inputs: torch.Tensor = batch["input"].to(self.device)
                targets: torch.Tensor = batch["target"].to(self.device)

                outputs: torch.Tensor = self.model(inputs)
                if self.experiment_type == "synthetic_arithmetic":
                    loss = self.main_loss_fn(outputs.squeeze(), targets)
                else:
                    loss = self.main_loss_fn(outputs, targets)
                total_loss += loss.item()

                batch_acc = self._calculate_accuracy(outputs, targets)
                total_accuracy += batch_acc
                batch_count += 1

        avg_loss: float = total_loss / batch_count if batch_count > 0 else 0.0
        avg_accuracy: float = total_accuracy / batch_count if batch_count > 0 else 0.0

        logging.info(f"Validation - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

        return {"validation_loss": avg_loss, "validation_accuracy": avg_accuracy}

# End of trainer.py
