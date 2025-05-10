"""
utils.py

------------------------------------------------------------
Logic Analysis for utils.py

Purpose:
  This module serves as the shared utility layer across the codebase. Its responsibilities include:
    • Reading and parsing the central configuration (from config.yaml)
    • Setting up a consistent logging system for all modules (to track progress and errors)
    • Initializing random seeds for reproducibility (across Python’s random, numpy, and torch)
    • Providing helper functions for saving and loading checkpoints (for models and optimizers)
    • Offering standardized plotting functions to create and save figures (used by the Evaluation module and Experiment Manager)
    • Including miscellaneous utilities (e.g., directory creation, converting time strings to seconds)

1. Configuration Management:
   - load_config(filepath: str = "config.yaml") reads the YAML file "config.yaml" and returns the configuration dict.
   - The config contains sections for training experiments, experiment_manager parameters, and general settings.

2. Logging Setup:
   - setup_logging(log_level: int = logging.INFO, log_file: str = "experiment.log") initializes Python's logging for both console and file output, ensuring unified logging.
  
3. Random Seed Initialization:
   - set_random_seed(seed: int) sets the seed for Python’s random module, NumPy, and Torch (both CPU and GPU if available).

4. Checkpoint Save/Load Helpers:
   - save_checkpoint(model, optimizer, epoch, checkpoint_dir="checkpoints", scheduler=None) saves the model and optimizer states to a checkpoint file.
   - load_checkpoint(file_path, model, optimizer, scheduler=None) loads the checkpoint and restores states.

5. Common Plot Functions:
   - save_plot(fig, filename) ensures the "figures/" folder exists and saves the provided matplotlib figure with high resolution (300 dpi).
  
6. Miscellaneous Utilities:
   - create_directory(path) creates a directory if it does not already exist.
   - convert_time_to_seconds(time_str) converts a time string like "1h", "30m", or "45s" to seconds.
  
This module is the backbone for reproducibility and consistency across the experimental pipeline.
------------------------------------------------------------
"""

import os
import logging
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt


def load_config(filepath: str = "config.yaml") -> Dict[str, Any]:
    """
    Load and parse the YAML configuration file.

    Args:
        filepath (str): Path to the YAML configuration file. Defaults to "config.yaml".

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    try:
        with open(filepath, "r") as file:
            config: Dict[str, Any] = yaml.safe_load(file)
        logging.info(f"Configuration loaded from '{filepath}'.")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration file '{filepath}': {e}")
        return {}


def setup_logging(log_level: int = logging.INFO, log_file: str = "experiment.log") -> None:
    """
    Configure the logging system to output messages to both console and a file.

    Args:
        log_level (int): Logging level. Defaults to logging.INFO.
        log_file (str): Path to the log file. Defaults to "experiment.log".
    """
    # Ensure the log directory is created (if file is in a subdirectory)
    log_dir: str = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Reset any existing handlers
    logging.root.handlers = []
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a")
        ]
    )
    logging.info(f"Logging is set up at level {logging.getLevelName(log_level)}. Log file: '{log_file}'.")


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and Torch.

    Args:
        seed (int): Seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed} for reproducibility.")


def create_directory(path: str) -> None:
    """
    Create a directory if it does not already exist.

    Args:
        path (str): Directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Directory created: {path}")
    else:
        logging.debug(f"Directory already exists: {path}")


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, checkpoint_dir: str = "checkpoints",
                    scheduler: Optional[Any] = None) -> str:
    """
    Save a checkpoint with the model state, optimizer state, scheduler state (if provided), and current epoch.

    Args:
        model (torch.nn.Module): The model to checkpoint.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch number.
        checkpoint_dir (str): Directory to save checkpoints. Defaults to "checkpoints".
        scheduler (Optional[Any]): Learning rate scheduler (if any).

    Returns:
        str: Path to the saved checkpoint file.
    """
    create_directory(checkpoint_dir)
    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    checkpoint_path: str = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    try:
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved at {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error saving checkpoint at {checkpoint_path}: {e}")
    return checkpoint_path


def load_checkpoint(file_path: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler: Optional[Any] = None) -> int:
    """
    Load a checkpoint from a file and restore states for the model, optimizer, and scheduler.

    Args:
        file_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        scheduler (Optional[Any]): Learning rate scheduler (if available).

    Returns:
        int: The epoch number stored in the checkpoint.
    """
    try:
        checkpoint: Dict[str, Any] = torch.load(file_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint.get("model_state_dict", {}))
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint.get("scheduler_state_dict"))
        epoch: int = checkpoint.get("epoch", 0)
        logging.info(f"Checkpoint loaded from {file_path} (epoch {epoch}).")
        return epoch
    except Exception as e:
        logging.error(f"Error loading checkpoint from {file_path}: {e}")
        return 0


def save_plot(fig: plt.Figure, filename: str) -> None:
    """
    Save a matplotlib figure to the "figures" directory with high resolution.

    Args:
        fig (plt.Figure): Matplotlib figure to save.
        filename (str): Filename to save the figure (e.g., "plot.png").
    """
    figures_dir: str = "figures"
    create_directory(figures_dir)
    file_path: str = os.path.join(figures_dir, filename)
    try:
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Plot saved as {file_path}")
    except Exception as e:
        logging.error(f"Error saving plot to {file_path}: {e}")


def convert_time_to_seconds(time_str: str) -> int:
    """
    Convert a time duration string to seconds.
    Supported formats:
      - "1h" for 1 hour
      - "30m" for 30 minutes
      - "45s" for 45 seconds

    Args:
        time_str (str): Time duration string.

    Returns:
        int: Equivalent duration in seconds.
    """
    time_str = time_str.strip().lower()
    try:
        if time_str.endswith("h"):
            seconds: int = int(float(time_str[:-1]) * 3600)
        elif time_str.endswith("m"):
            seconds = int(float(time_str[:-1]) * 60)
        elif time_str.endswith("s"):
            seconds = int(float(time_str[:-1]))
        else:
            # If no suffix, assume seconds.
            seconds = int(float(time_str))
        logging.info(f"Converted time '{time_str}' to {seconds} seconds.")
        return seconds
    except Exception as e:
        logging.error(f"Error converting time string '{time_str}' to seconds: {e}")
        return 0


# Additional utility: write JSON logs (if needed in future)
def write_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Write a dictionary to a JSON file.

    Args:
        data (Dict[str, Any]): Data dictionary to write.
        filepath (str): Path to the JSON file.
    """
    import json
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"JSON data written to {filepath}")
    except Exception as e:
        logging.error(f"Error writing JSON data to {filepath}: {e}")
