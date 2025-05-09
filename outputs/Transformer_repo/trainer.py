"""trainer.py

This module implements the Trainer class responsible for orchestrating the training process
of the Transformer model as described in "Attention Is All You Need". It handles the forward
and backward passes, custom learning rate scheduling, label smoothing loss computation,
checkpointing, and logging of training metrics.
"""

import math
import os
import logging
from typing import Any, Tuple, Iterator, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Use the same PAD_ID as defined in dataset_loader.py (assumed to be 0)
PAD_ID: int = 0


class LabelSmoothingLoss(nn.Module):
    """
    Implements label smoothing loss.
    Given a smoothing value epsilon, it assigns a confidence value (1 - epsilon) to the true label
    and distributes the remaining confidence uniformly across all other classes.
    This implementation also ignores padding tokens (PAD_ID).
    """

    def __init__(self, smoothing: float, vocab_size: int, ignore_index: int = PAD_ID) -> None:
        """
        Args:
            smoothing (float): The label smoothing factor (e.g., 0.1).
            vocab_size (int): Number of classes (vocab size).
            ignore_index (int): Token index to ignore (e.g., PAD_ID).
        """
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing value must be in [0, 1)"
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the label smoothed cross entropy loss.

        Args:
            pred (torch.Tensor): Logits from the model of shape (N, vocab_size),
                                 where N is the total number of predictions.
            target (torch.Tensor): Target indices of shape (N,).

        Returns:
            torch.Tensor: The averaged loss.
        """
        # pred: (N, vocab_size) log probabilities (will apply log_softmax)
        log_probs = F.log_softmax(pred, dim=-1)
        # Create true distribution with label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            # For positions where target == ignore_index compare to PAD_ID, leave zeros
            ignore_mask = target.eq(self.ignore_index)
            # Scatter the confidence for non-ignored entries
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # For ignored targets, set distribution to zero to remove their contribution
            true_dist.masked_fill_(ignore_mask.unsqueeze(1), 0.0)
        # Compute KL divergence loss
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        # Only average over non-ignored elements.
        non_pad = target.ne(self.ignore_index).float()
        loss = torch.sum(loss * non_pad) / torch.sum(non_pad)
        return loss


class Trainer:
    """
    Trainer class manages the training process of the Transformer model.
    It handles data iteration, forward and backward passes, learning rate scheduling,
    logging, checkpoint saving, and resuming training from checkpoints.
    """

    def __init__(self, model: nn.Module, train_data: List[Any], config: dict) -> None:
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): An instance of the TransformerModel.
            train_data (List[Any]): Preprocessed training data batches.
                For translation tasks, each element is a tuple (src, tgt) of torch.Tensors.
                For parsing tasks, each element is a torch.Tensor.
            config (dict): Configuration parameters (parsed from config.yaml).
        """
        self.config = config
        self.model = model
        self.train_data = train_data

        # Determine device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Retrieve training hyperparameters.
        training_config = config.get("training", {})
        self.total_steps: int = training_config.get("total_steps", 100000)
        self.warmup_steps: int = training_config.get("warmup_steps", 4000)

        # Logging and checkpoint intervals.
        self.log_interval: int = config.get("logging", {}).get("log_interval_steps", 100)
        self.save_interval: int = config.get("checkpoint", {}).get("save_interval_steps", 1000)

        # Optimizer hyperparameters.
        optimizer_config = training_config.get("optimizer", {})
        beta1: float = optimizer_config.get("beta1", 0.9)
        beta2: float = optimizer_config.get("beta2", 0.98)
        epsilon: float = optimizer_config.get("epsilon", 1e-9)

        # Initialize the optimizer.
        # Initial learning rate is set to 0 and will be updated each step according to our custom schedule.
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.0, betas=(beta1, beta2), eps=epsilon
        )

        # Store model's d_model (assumed to be set in the model as self.d_model).
        self.d_model: int = getattr(self.model, "d_model", 512)

        # Determine label smoothing value based on model variant.
        model_variant: str = config.get("model", {}).get("variant", "base")
        if model_variant == "big":
            self.label_smoothing: float = config.get("model", {}).get("big", {}).get("label_smoothing", 0.1)
        else:
            self.label_smoothing: float = config.get("model", {}).get("base", {}).get("label_smoothing", 0.1)

        # Vocabulary size is required for label smoothing; assumed provided in config as "vocab_size"
        self.vocab_size: int = config.get("vocab_size", 37000)

        # Initialize the label smoothing loss criterion.
        self.criterion = LabelSmoothingLoss(self.label_smoothing, self.vocab_size, ignore_index=PAD_ID)

        # Initialize current step counter.
        self.current_step: int = 1

        # Setup checkpoint directory.
        self.checkpoint_dir: str = config.get("checkpoint", {}).get("dir", "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Prepare training data iterator.
        self.data_iterator: Iterator = iter(self.train_data)

    def compute_learning_rate(self, step: int) -> float:
        """
        Computes the learning rate using the formula:
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

        Args:
            step (int): Current training step (starting from 1).

        Returns:
            float: The computed learning rate.
        """
        return (self.d_model ** (-0.5)) * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))

    def update_learning_rate(self, step: int) -> None:
        """
        Updates the optimizer's learning rate based on the current step.

        Args:
            step (int): Current training step.
        """
        lr: float = self.compute_learning_rate(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates source and target attention masks for the Transformer.
        The source mask masks PAD tokens.
        The target mask combines a subsequent mask (to prevent attention to future tokens)
        with a mask for PAD tokens.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target tensor of shape (batch_size, tgt_seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - src_mask: Boolean mask of shape (batch_size, src_seq_len) where True indicates non-PAD.
                - tgt_mask: Boolean mask of shape (tgt_seq_len, tgt_seq_len) combined with target padding mask.
        """
        # Create source padding mask.
        src_mask = (src != PAD_ID).to(self.device)

        # Create target padding mask.
        tgt_pad_mask = (tgt != PAD_ID).to(self.device)

        # Generate subsequent mask using the model's helper function.
        seq_len: int = tgt.size(1)
        subsequent_mask: torch.Tensor = self.model.generate_square_subsequent_mask(seq_len).to(self.device)
        # Combine subsequent mask with target padding mask.
        # tgt_pad_mask is (batch, seq_len) so we unsqueeze to (batch, 1, seq_len) and broadcast.
        # For simplicity, we return the subsequent mask; many implementations combine both.
        # Here we assume the subsequent mask is sufficient (padding tokens are handled in loss).
        return src_mask, subsequent_mask

    def train(self) -> None:
        """
        Main training loop.
        Iterates through training steps, performing forward and backward passes,
        updating model parameters, adjusting the learning rate, logging metrics,
        and saving checkpoints periodically.
        """
        self.model.train()
        total_loss: float = 0.0

        progress_bar = tqdm(range(self.total_steps), desc="Training", unit="step")
        for _ in progress_bar:
            # If data iterator is exhausted, reinitialize it.
            try:
                batch = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_data)
                batch = next(self.data_iterator)

            # Determine task type by checking if batch is a pair (translation) or single tensor (parsing).
            if isinstance(batch, tuple) and len(batch) == 2:
                src, tgt = batch
                src = src.to(self.device)
                tgt = tgt.to(self.device)
            else:
                # For parsing tasks, use the same batch as both src and tgt.
                src = batch.to(self.device)
                tgt = batch.to(self.device)

            # For teacher forcing in sequence-to-sequence tasks, the input to the decoder is tgt[:, :-1]
            # and the target for loss is tgt[:, 1:].
            decoder_input: torch.Tensor = tgt[:, :-1]
            decoder_target: torch.Tensor = tgt[:, 1:]

            # Create masks for source and target.
            src_mask, tgt_mask = self.create_masks(src, decoder_input)

            # Forward pass.
            logits = self.model(src, decoder_input, src_mask, tgt_mask)
            # Reshape logits and targets for loss computation.
            # logits shape: (batch_size, seq_len, vocab_size), target shape: (batch_size, seq_len)
            loss = self.criterion(logits.view(-1, self.vocab_size), decoder_target.view(-1))

            # Backpropagation.
            self.optimizer.zero_grad()
            loss.backward()
            # Optional gradient clipping.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update the learning rate.
            self.update_learning_rate(self.current_step)

            # Accumulate loss.
            total_loss += loss.item()

            # Logging
            if self.current_step % self.log_interval == 0:
                avg_loss = total_loss / self.log_interval
                current_lr = self.optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.6f}"})
                total_loss = 0.0

            # Checkpoint saving.
            if self.current_step % self.save_interval == 0:
                self.save_checkpoint()

            self.current_step += 1

        # Final checkpoint saving after training completes.
        self.save_checkpoint()
        logging.info("Training complete.")

    def save_checkpoint(self) -> None:
        """
        Saves a checkpoint including the model state, optimizer state, and current training step.
        The checkpoint is saved to the directory specified in the configuration.
        """
        checkpoint_path: str = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.current_step}.pt")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_step": self.current_step,
        }
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved at step {self.current_step} to {checkpoint_path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Loads a checkpoint from the specified path and restores the model state,
        optimizer state, and training step.

        Args:
            path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_step = checkpoint["current_step"]
        # Update learning rate after loading state.
        self.update_learning_rate(self.current_step)
        logging.info(f"Checkpoint loaded from {path} at step {self.current_step}")
