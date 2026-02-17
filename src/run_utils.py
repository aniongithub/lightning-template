"""Utilities for running training and evaluation workflows."""

import torch
from pathlib import Path


def find_best_checkpoint(model_class, litmodule_class, checkpoint_dir="./checkpoints"):
    """Find and load the latest checkpoint with model instantiation.
    
    Args:
        model_class: The model class to instantiate (e.g., SimpleDenseNet)
        litmodule_class: The LightningModule class (e.g., MNISTLitModule)
        checkpoint_dir: Directory containing checkpoints (default: ./checkpoints)
    
    Returns:
        Tuple of (model, error_message) where model is loaded LightningModule or None if not found
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None, f"Checkpoint directory not found: {checkpoint_dir}"
    
    checkpoints = list(checkpoint_path.glob("*.ckpt"))
    if not checkpoints:
        return None, "No checkpoints found. Train the model first."
    
    best_checkpoint = checkpoints[-1]  # Load latest checkpoint
    print(f"Loading checkpoint: {best_checkpoint}")
    
    try:
        # Instantiate model with fresh net, then load trained weights
        net = model_class()
        model = litmodule_class(net=net)
        state_dict = torch.load(best_checkpoint, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        return model, None
    except Exception as e:
        return None, f"Error loading checkpoint: {str(e)}"
