"""
train_simple.py

Training entry point for the SimpleSCMGNN operation-prediction model.

This script trains a lightweight graph neural network on SCM decomposition
states. Each input graph represents a partial decomposition trace, and the
model predicts the next operation type (`op`) among four classes. The training
pipeline includes sample-level train/validation splitting, optimized PyG data
loading, optional mixed-precision training, gradient clipping, early stopping,
checkpoint saving, and CSV-based experiment logging.

Main responsibilities:
    1. Load pre-defined train/test target splits.
    2. Construct SCMGraphDataset objects for graph generation.
    3. Split the training set into train/validation subsets at the sample level.
    4. Train SimpleSCMGNN for operation classification.
    5. Save the best checkpoint, training history, and final test results.

Research note:
    The validation split is performed by original sample index rather than by
    graph index. This avoids data leakage caused by placing different prefix
    graphs from the same SCM decomposition sample into both train and validation
    sets.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from gnn_model_simple import SimpleSCMGNN
from graph_dataset import SCMGraphDataset

from utils import (
    create_optimizer_and_scheduler,
    get_current_lr,
    create_optimized_dataloader,
    get_optimal_workers,
    set_seed,
)


# Fix random seeds for reproducible experiments.
set_seed(seed=42)


def compute_simple_loss(pred, target, config):
    """
    Compute the operation-classification loss.

    This simplified training setting focuses only on predicting the next
    operation type. Therefore, the objective is a standard cross-entropy loss
    over four operation classes.

    Args:
        pred (dict):
            Model output dictionary. Expected key:
                - 'op': logits with shape [batch_size, 4].
        target (dict):
            Target dictionary. Expected key:
                - 'y_op': operation labels with shape [batch_size].
        config (dict):
            Training configuration. Included for interface consistency with
            other training scripts, even though this simple loss does not use
            additional weights.

    Returns:
        tuple:
            - total_loss (torch.Tensor): scalar training loss.
            - loss_dict (dict): detached scalar components for logging.
    """
    device = pred['op'].device

    # Cross-entropy loss for 4-class operation prediction.
    op_loss = nn.functional.cross_entropy(
        pred['op'],
        target['y_op'],
        reduction='mean',
    )

    total_loss = op_loss

    loss_dict = {
        'op_loss': op_loss.item(),
    }

    return total_loss, loss_dict


def prepare_simple_target(data):
    """
    Extract operation labels from a PyG batch.

    Args:
        data:
            PyTorch Geometric Batch object produced by SCMGraphDataset.

    Returns:
        dict:
            Dictionary containing y_op labels with shape [batch_size].
    """
    batch_size = data.num_graphs
    device = data.x.device

    # The dataset normally provides y_op. The fallback keeps the function safe
    # during debugging with synthetic data.
    if hasattr(data, 'y_op'):
        y_op = data.y_op.view(batch_size)
    else:
        y_op = torch.zeros(batch_size, dtype=torch.long, device=device)

    return {
        'y_op': y_op,
    }


def compute_top1_accuracy(pred, target):
    """
    Compute top-1 classification accuracy.

    Args:
        pred (torch.Tensor): predicted class indices.
        target (torch.Tensor): ground-truth class indices.

    Returns:
        float: fraction of correctly classified examples.
    """
    if len(pred) == 0 or len(target) == 0:
        return 0.0

    correct = (pred == target).sum().item()
    total = len(target)

    return correct / total if total > 0 else 0.0


def train_one_epoch(model, loader, optimizer, device, config, scaler=None):
    """
    Train the model for one epoch.

    This routine supports both standard FP32 training and optional automatic
    mixed precision (AMP). It also includes gradient accumulation, gradient
    clipping, and defensive checks for non-finite losses or gradients.

    Args:
        model (nn.Module): SimpleSCMGNN model.
        loader (DataLoader): training data loader.
        optimizer: PyTorch optimizer.
        device: target computation device.
        config (dict): training hyperparameters.
        scaler (torch.cuda.amp.GradScaler, optional): AMP gradient scaler.

    Returns:
        tuple:
            - avg_loss (float): average total loss over processed samples.
            - dict: averaged loss components.
    """
    model.train()

    total_loss = 0.0
    total_op_loss = 0.0
    total_samples = 0
    skipped_batches = 0

    accumulation_steps = config.get('accumulation_steps', 1)
    grad_clip = config.get('grad_clip', 5.0)

    optimizer.zero_grad()

    for batch_idx, data in enumerate(tqdm(loader, desc="Training", leave=False)):
        try:
            # Move the PyG batch to GPU/CPU. non_blocking=True can improve
            # throughput when pinned memory is enabled in the DataLoader.
            data = data.to(device, non_blocking=True)

            if scaler is not None:
                # AMP branch: reduce memory usage and speed up training on CUDA.
                with torch.cuda.amp.autocast():
                    pred = model(data)
                    target = prepare_simple_target(data)
                    loss, loss_dict = compute_simple_loss(pred, target, config)

                    if not torch.isfinite(loss):
                        print(f"\n⚠️ Warning: Invalid loss at batch {batch_idx}: {loss.item()}")
                        skipped_batches += 1
                        optimizer.zero_grad()
                        continue

                    # Normalize the loss when using gradient accumulation.
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                # Detect non-finite gradients before the optimizer update.
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"\n⚠️ Warning: NaN gradient in {name} at batch {batch_idx}")
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    skipped_batches += 1
                    optimizer.zero_grad()
                    continue

                # Update parameters after the configured number of accumulated
                # mini-batches.
                if (batch_idx + 1) % accumulation_steps == 0:
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            else:
                # Standard FP32 training branch.
                pred = model(data)
                target = prepare_simple_target(data)
                loss, loss_dict = compute_simple_loss(pred, target, config)

                if not torch.isfinite(loss):
                    print(f"\n⚠️ Warning: Invalid loss at batch {batch_idx}: {loss.item()}")
                    skipped_batches += 1
                    optimizer.zero_grad()
                    continue

                loss = loss / accumulation_steps
                loss.backward()

                # Check gradients explicitly to avoid corrupting the optimizer
                # state with NaN/Inf values.
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"\n⚠️ Warning: NaN gradient in {name} at batch {batch_idx}")
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    skipped_batches += 1
                    optimizer.zero_grad()
                    continue

                if (batch_idx + 1) % accumulation_steps == 0:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                    optimizer.step()
                    optimizer.zero_grad()

            batch_size = data.num_graphs
            total_loss += loss.item() * accumulation_steps * batch_size
            total_op_loss += loss_dict['op_loss'] * batch_size
            total_samples += batch_size

        except Exception as e:
            # Keep long experiments running even if a rare malformed batch appears.
            print(f"\n❌ Error at batch {batch_idx}: {e}")
            skipped_batches += 1
            optimizer.zero_grad()
            continue

    # If the epoch ends before completing a full accumulation cycle, apply the
    # remaining gradients once.
    if (batch_idx + 1) % accumulation_steps != 0:
        if grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_op_loss = total_op_loss / total_samples
    else:
        avg_loss = float('inf')
        avg_op_loss = float('inf')

    if skipped_batches > 0:
        print(f"\n⚠️ Skipped {skipped_batches} batches due to invalid loss/gradients")

    return avg_loss, {'op_loss': avg_op_loss}


def evaluate(model, loader, device, config, use_amp=False):
    """
    Evaluate the model on a data split.

    Args:
        model (nn.Module): trained or partially trained model.
        loader (DataLoader): validation/test data loader.
        device: computation device.
        config (dict): training configuration.
        use_amp (bool): whether to use AMP during inference.

    Returns:
        dict:
            Metrics including loss, op_loss, and op_acc.
    """
    model.eval()

    total_loss = 0.0
    total_op_loss = 0.0
    total_samples = 0

    all_op_preds = []
    all_op_labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            data = data.to(device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(data)
            else:
                pred = model(data)

            target = prepare_simple_target(data)
            loss, loss_dict = compute_simple_loss(pred, target, config)

            batch_size = data.num_graphs
            total_loss += loss.item() * batch_size
            total_op_loss += loss_dict['op_loss'] * batch_size
            total_samples += batch_size

            all_op_preds.append(pred['op'].argmax(-1).cpu())
            all_op_labels.append(target['y_op'].cpu())

    all_op_preds = torch.cat(all_op_preds) if all_op_preds else torch.tensor([])
    all_op_labels = torch.cat(all_op_labels) if all_op_labels else torch.tensor([])

    metrics = {
        'loss': total_loss / max(1, total_samples),
        'op_loss': total_op_loss / max(1, total_samples),
    }

    if total_samples > 0 and len(all_op_preds) > 0:
        metrics['op_acc'] = compute_top1_accuracy(all_op_preds, all_op_labels)
    else:
        metrics['op_acc'] = 0.0

    return metrics


def create_data_splits_by_sample(dataset, val_ratio=0.1, random_state=42):
    """
    Create train/validation graph indices using sample-level splitting.

    SCMGraphDataset expands each original sample into multiple prefix graphs.
    Splitting directly at the graph level would leak information because graphs
    derived from the same original decomposition trace could appear in both
    training and validation sets. This function first splits unique sample IDs,
    then maps them back to graph indices.

    Args:
        dataset (SCMGraphDataset): graph dataset with an index_map attribute.
        val_ratio (float): fraction of original samples used for validation.
        random_state (int): random seed for reproducibility.

    Returns:
        tuple[list[int], list[int]]:
            Graph indices for training and validation subsets.
    """
    unique_sample_indices = list(set(idx for idx, k in dataset.index_map))

    train_sample_indices, val_sample_indices = train_test_split(
        unique_sample_indices,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True,
    )

    train_graph_indices = []
    val_graph_indices = []

    for graph_idx, (sample_idx, k) in enumerate(dataset.index_map):
        if sample_idx in train_sample_indices:
            train_graph_indices.append(graph_idx)
        elif sample_idx in val_sample_indices:
            val_graph_indices.append(graph_idx)

    print(f"📊 Data splits:")
    print(f"  Training samples: {len(train_sample_indices)} → {len(train_graph_indices)} graphs")
    print(f"  Validation samples: {len(val_sample_indices)} → {len(val_graph_indices)} graphs")

    return train_graph_indices, val_graph_indices


def train_model(model, train_loader, val_loader, test_loader, device, config):
    """
    Run the full training, validation, checkpointing, and testing pipeline.

    The best model is selected by validation loss. Training history is saved as
    CSV for later plotting or reporting, and the best checkpoint is saved for
    inference.

    Args:
        model (nn.Module): SimpleSCMGNN model.
        train_loader (DataLoader): training graph loader.
        val_loader (DataLoader): validation graph loader.
        test_loader (DataLoader): held-out test graph loader.
        device: computation device.
        config (dict): experiment hyperparameters.

    Returns:
        tuple:
            - test_results (dict): final test metrics.
            - best_model_state (dict): best model state dictionary.
    """
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    use_amp = config.get('use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    if use_amp:
        print("🎯 Using Mixed Precision Training (AMP)")

    best_val_loss = float('inf')
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    wait = 0
    train_history = []
    current_lr = get_current_lr(optimizer)

    print(f"\n{'=' * 80}")
    print("Training SimpleSCMGNN - Op Prediction Only")
    print(f"LR Strategy: {config['lr_scheduler']}")
    print(f"Device: {device}")
    print(f"{'=' * 80}")

    for epoch in range(1, config['epochs'] + 1):
        try:
            train_loss, train_components = train_one_epoch(
                model, train_loader, optimizer, device, config, scaler
            )

            # Evaluate on both train and validation splits to track overfitting.
            train_metrics = evaluate(model, train_loader, device, config, use_amp=False)
            val_metrics = evaluate(model, val_loader, device, config, use_amp=False)

            current_lr = get_current_lr(optimizer)

            # Scheduler step is performed once per epoch.
            if config['lr_scheduler'] != 'fixed':
                scheduler.step()

            history_entry = {
                'epoch': epoch,
                'learning_rate': current_lr,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_op_loss': train_metrics['op_loss'],
                'val_op_loss': val_metrics['op_loss'],
                'train_op_acc': train_metrics['op_acc'],
                'val_op_acc': val_metrics['op_acc'],
            }
            train_history.append(history_entry)

            # Save the checkpoint with the best validation loss.
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                wait = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'val_metrics': val_metrics,
                    'config': config,
                }, "./model_results/best_model_simple.pth")

                print(
                    f"Epoch {epoch:3d} [BEST] | "
                    f"LR: {current_lr:.2e} | "
                    f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                    f"Op Acc: {train_metrics['op_acc']:.4f}/{val_metrics['op_acc']:.4f}"
                )
            else:
                wait += 1
                if wait >= config['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % config['print_every'] == 0 and wait > 0:
                print(
                    f"Epoch {epoch:3d}        | "
                    f"LR: {current_lr:.2e} | "
                    f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                    f"Op Acc: {train_metrics['op_acc']:.4f}/{val_metrics['op_acc']:.4f} | "
                    f"Wait: {wait}/{config['patience']}"
                )

        except Exception as e:
            # Continue training when a recoverable epoch-level exception occurs.
            print(f"❌ Error at epoch {epoch}: {e}")
            print("Continuing with next epoch...")
            continue

    print(f"\n{'=' * 60}")
    print("Final Evaluation on Test Set")
    print(f"{'=' * 60}")

    try:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model from epoch {best_epoch}")
    except Exception as e:
        print(f"❌ Error loading best model: {e}")
        print("Using current model for testing")

    test_metrics = evaluate(model, test_loader, device, config, use_amp=use_amp)

    print(f"\n📊 Test Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Op Loss: {test_metrics['op_loss']:.4f}")
    print(f"  Op Accuracy: {test_metrics['op_acc']:.4f}")

    # Persist epoch-level metrics for later analysis and plotting.
    history_df = pd.DataFrame(train_history)
    os.makedirs("./training_history", exist_ok=True)
    history_df.to_csv("./training_history/training_history_simple.csv", index=False)

    test_results = {
        'final_epoch': len(train_history),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_lr': current_lr,
        **test_metrics,
    }

    test_results_df = pd.DataFrame([test_results])
    test_results_df.to_csv("./model_results/test_results_simple.csv", index=False)

    print("\n✅ Training history saved: training_history/training_history_simple.csv")
    print("✅ Test results saved: model_results/test_results_simple.csv")
    print(f"✅ Best model: epoch {best_epoch}, val_loss: {best_val_loss:.4f}")

    return test_results, best_model_state


def main():
    """
    Configure and launch the SimpleSCMGNN training experiment.

    The current configuration trains a GATv2-based model for OP-only prediction
    using 199-dimensional node features and 12-dimensional edge features.
    """
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")

    # Centralized experiment configuration. These values are stored in the
    # checkpoint to make later inference and reproduction easier.
    config = {
        'batch_size': 256,
        'accumulation_steps': 1,
        'epochs': 800,
        'patience': 20,
        'print_every': 10,
        'use_amp': False,

        # Model architecture.
        'node_in_dim': 199,
        'edge_in_dim': 12,
        'hidden_dim': 128,
        'num_heads': 8,
        'dropout': 0.1,
        'conv_type': 'gatv2',
        'num_gnn_layers': 5,

        # Data split and optimization.
        'val_ratio': 0.1,
        'random_state': 42,
        'lr': 2e-4,
        'weight_decay': 1e-3,
        'grad_clip': 5.0,

        # Learning-rate schedule.
        'lr_scheduler': 'cosine_warm_restarts',
        'min_lr': 1e-6,
        'restart_t0': 60,
        'restart_t_mult': 2,
        'warmup_steps': 100,
        'warmup_start_factor': 0.1,
    }

    os.makedirs("./model_results", exist_ok=True)
    os.makedirs("./training_history", exist_ok=True)

    print("\n📋 Model Configuration (Simple - OP Only):")
    print(f"  Node Input Dim: {config['node_in_dim']}")
    print(f"  Hidden Dim: {config['hidden_dim']}")
    print(f"  GNN Layers: {config['num_gnn_layers']}")
    print(f"  Conv Type: {config['conv_type']}")
    print("  Task: Op prediction only (4 classes)")

    print("\n📦 Loading dataset splits...")
    from data_split import load_split_targets
    train_targets, test_targets = load_split_targets()

    print("\n📦 Loading training dataset...")
    train_dataset = SCMGraphDataset(
        "./data/split/train_data.json",
        max_prefix_len=11,
        split_type='train',
        train_targets=train_targets,
        test_targets=test_targets,
    )
    print(f"✅ Training dataset: {len(train_dataset.samples)} samples → {len(train_dataset)} graphs")

    print("\n📦 Loading test dataset...")
    test_dataset = SCMGraphDataset(
        "./data/split/test_data.json",
        max_prefix_len=11,
        split_type='test',
        train_targets=train_targets,
        test_targets=test_targets,
    )
    print(f"✅ Test dataset: {len(test_dataset.samples)} samples → {len(test_dataset)} graphs")

    # Create validation split from the training dataset without leaking graphs
    # from the same original sample across splits.
    train_indices, val_indices = create_data_splits_by_sample(
        train_dataset,
        val_ratio=config['val_ratio'],
        random_state=config['random_state'],
    )

    print(f"🔧 Creating optimized data loaders with {get_optimal_workers()} workers...")
    train_loader = create_optimized_dataloader(
        Subset(train_dataset, train_indices),
        batch_size=config['batch_size'],
        shuffle=True,
        is_train=True,
    )
    val_loader = create_optimized_dataloader(
        Subset(train_dataset, val_indices),
        batch_size=config['batch_size'],
        shuffle=False,
        is_train=False,
    )
    test_loader = create_optimized_dataloader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        is_train=False,
    )

    print("\n📊 Final Data Loaders (Graphs):")
    print(f"  Training: {len(train_loader.dataset)} graphs")
    print(f"  Validation: {len(val_loader.dataset)} graphs")
    print(f"  Test: {len(test_loader.dataset)} graphs")

    model = SimpleSCMGNN(
        node_in_dim=config['node_in_dim'],
        edge_in_dim=config['edge_in_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        conv_type=config['conv_type'],
        dropout=config['dropout'],
        num_gnn_layers=config['num_gnn_layers'],
    ).to(device)

    print(f"\n📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    test_results, best_model_state = train_model(
        model, train_loader, val_loader, test_loader, device, config
    )

    print("\n🎉 Training Completed!")
    print("📁 Generated files:")
    print("   - model_results/best_model_simple.pth")
    print("   - training_history/training_history_simple.csv")
    print("   - model_results/test_results_simple.csv")


if __name__ == "__main__":
    main()
