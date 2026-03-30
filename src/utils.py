"""Utility functions for GPT model training and evaluation."""

import torch
import os
from typing import Tuple


def load_text_data(file_path: str) -> str:
    """Load text data from file.
    
    Args:
        file_path: Path to text file
        
    Returns:
        String containing file contents
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_checkpoint(model, optimizer, step: int, checkpoint_dir: str = ".") -> None:
    """Save model and optimizer state.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        step: Current training step
        checkpoint_dir: Directory to save checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, model, optimizer) -> int:
    """Load model and optimizer state from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load into
        optimizer: PyTorch optimizer to load into
        
    Returns:
        Step number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from step {step}")
    return step


def count_parameters(model) -> int:
    """Count total trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size(model) -> float:
    """Estimate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in megabytes
    """
    param_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2)
    return param_size
