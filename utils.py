"""
utils.py

Utility functions for reproducible GNN training and efficient data loading.

This module provides:
    1. Random seed control for reproducible experiments.
    2. Optimizer and learning-rate scheduler construction.
    3. Learning-rate inspection.
    4. Automatic DataLoader worker selection.
    5. Optimized PyTorch Geometric DataLoader creation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.loader import DataLoader
from collections import Counter
from tqdm import tqdm
import numpy as np
import random

from torch.optim.lr_scheduler import SequentialLR, LinearLR


def set_seed(seed=42):
    """
    Set random seeds across major libraries to improve experiment reproducibility.

    This function controls randomness from:
        - PyTorch CPU operations
        - PyTorch CUDA operations
        - NumPy
        - Python's built-in random module

    It also configures cuDNN to use deterministic behavior. This may slightly
    reduce performance, but it helps ensure that repeated runs are more
    comparable.

    Args:
        seed (int):
            Random seed used throughout the experiment.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Force deterministic cuDNN behavior for reproducibility.
    torch.backends.cudnn.deterministic = True

    # Disable cuDNN benchmark mode because it may introduce nondeterministic
    # algorithm selection depending on input shapes.
    torch.backends.cudnn.benchmark = False


# ==================== Optimizer and Scheduler Utilities ====================

def create_optimizer_and_scheduler(model, config):
    """
    Create the optimizer and learning-rate scheduler for model training.

    The optimizer is AdamW, which is commonly used for neural network training
    because it decouples weight decay from gradient-based parameter updates.

    The scheduler supports:
        - Linear warmup
        - Cosine annealing
        - Cosine annealing with warm restarts
        - Step decay
        - Constant learning rate

    If warmup is enabled, a SequentialLR scheduler is used to combine:
        1. Linear warmup stage
        2. Main scheduler stage

    Args:
        model (torch.nn.Module):
            Model whose parameters will be optimized.

        config (dict):
            Training configuration dictionary. Expected keys may include:
                - lr
                - weight_decay
                - lr_scheduler
                - warmup_steps
                - warmup_start_factor
                - cosine_t_max
                - min_lr
                - restart_t0
                - restart_t_mult
                - step_size
                - step_gamma

    Returns:
        tuple:
            optimizer:
                AdamW optimizer.
            scheduler:
                Learning-rate scheduler.
    """

    # AdamW is used to improve regularization through decoupled weight decay.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 1. Optional linear warmup scheduler.
    # Warmup gradually increases the learning rate at the beginning of training,
    # which can stabilize optimization for deep GNN models.
    if config.get('warmup_steps', 0) > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=config.get('warmup_start_factor', 0.01),
            total_iters=config['warmup_steps']
        )
    else:
        warmup_scheduler = None

    # 2. Main learning-rate scheduler.
    if config['lr_scheduler'] == 'cosine':
        # Smoothly decays the learning rate following a cosine curve.
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['cosine_t_max'],
            eta_min=config['min_lr']
        )

    elif config['lr_scheduler'] == 'cosine_warm_restarts':
        # Periodically restarts the cosine schedule.
        # This can help the optimizer escape shallow local minima.
        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['restart_t0'],
            T_mult=config['restart_t_mult'],
            eta_min=config['min_lr']
        )

    elif config['lr_scheduler'] == 'step':
        # Decays the learning rate by a fixed factor every fixed number of steps.
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['step_gamma']
        )

    else:
        # Default fallback: keep learning rate constant.
        main_scheduler = optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0
        )

    # 3. Combine warmup and main scheduler if warmup is enabled.
    if warmup_scheduler:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config['warmup_steps']]
        )
    else:
        scheduler = main_scheduler

    return optimizer, scheduler


def get_current_lr(optimizer):
    """
    Return the current learning rate from the optimizer.

    This is useful for logging training progress, especially when using
    dynamic learning-rate schedulers.

    Args:
        optimizer (torch.optim.Optimizer):
            Optimizer object.

    Returns:
        float:
            Current learning rate of the first parameter group.
    """
    return optimizer.param_groups[0]['lr']


def get_optimal_workers():
    """
    Estimate a reasonable number of DataLoader worker processes.

    The heuristic leaves two CPU cores available for the operating system and
    other background tasks, while limiting the number of workers to avoid
    excessive multiprocessing overhead.

    Returns:
        int:
            Suggested number of DataLoader workers.
    """
    cpu_count = os.cpu_count()

    if cpu_count is None:
        # Fallback value when CPU count cannot be detected.
        return 4

    # Heuristic:
    #   - Use at least 2 workers.
    #   - Use at most 8 workers.
    #   - Reserve 2 CPU cores for the system and other tasks.
    optimal = max(2, min(8, cpu_count - 2))

    return optimal


def create_optimized_dataloader(dataset, batch_size, shuffle=True, is_train=True):
    """
    Create an optimized PyTorch Geometric DataLoader.

    This function configures DataLoader parameters to improve throughput during
    GNN training, especially when using GPU acceleration.

    Key optimizations:
        - Multiple worker processes for parallel data loading.
        - Pinned memory for faster CPU-to-GPU transfer.
        - Persistent workers to reduce worker startup overhead.
        - Prefetching to overlap data preparation with model computation.
        - Dropping the last incomplete batch for stable batch-level training.

    Args:
        dataset:
            PyTorch Geometric dataset or subset.

        batch_size (int):
            Number of graphs per batch.

        shuffle (bool):
            Whether to shuffle the dataset.

        is_train (bool):
            Whether the DataLoader is used for training.
            Training loaders use more workers and higher prefetching.

    Returns:
        DataLoader:
            Configured PyTorch Geometric DataLoader.
    """

    # Use more workers during training and fewer workers during evaluation.
    num_workers = get_optimal_workers() if is_train else 2

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,

        # Enables faster transfer to CUDA memory when using GPUs.
        pin_memory=True,

        # Keeps worker processes alive across epochs to reduce startup cost.
        persistent_workers=num_workers > 0,

        # Prefetch future batches to reduce data loading stalls.
        prefetch_factor=2 if is_train else 1,

        # Drop incomplete final batch for more stable training dynamics.
        drop_last=True,
    )

    return loader