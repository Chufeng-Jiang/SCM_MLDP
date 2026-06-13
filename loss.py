"""
loss.py

Training utilities for SCM decomposition prediction models.

This module provides helper functions for preparing supervised learning targets,
computing task-specific evaluation metrics, and constructing a weighted
multi-task loss for SCM decomposition prediction. The utilities are designed to
work with PyTorch Geometric batches produced by the SCM graph dataset.

The learning problem can be configured in two modes:
    1. simple mode:
        Predicts core decomposition attributes such as shift, operation type,
        and operand multiplier values.

    2. full mode:
        Extends the simple setting with reuse-aware supervision, including
        reuse pattern classification, reuse decision prediction, and reuse node
        selection for left and right operands.

The loss function combines classification and regression objectives. Operation
and shift prediction are treated as classification tasks, while multiplier
prediction is performed in log-space using Smooth L1 loss to reduce sensitivity
to large numeric ranges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def prepare_target_dict(data, mode='full'):
    """
    Convert a PyTorch Geometric batch into a dictionary of training targets.

    The dataset stores labels as attributes on each `Data` object. This function
    standardizes those attributes into a flat dictionary that can be consumed by
    the multi-task loss function.

    Args:
        data:
            A PyTorch Geometric `Data` or `Batch` object containing supervision
            fields such as `y_shift`, `y_op`, `y_left_mult`, and reuse labels.
        mode (str):
            Training mode. Use `'simple'` for core prediction targets only, or
            `'full'` to include reuse-related labels.

    Returns:
        dict:
            A dictionary mapping target names to 1D tensors with shape [B].
    """
    # Core classification targets shared by both simple and full modes.
    target = {
        'y_shift': data.y_shift.view(-1),
        'y_op':    data.y_op.view(-1),
    }

    # Optional multiplier regression targets. These represent the numeric values
    # of the left and right operands used in the next decomposition step.
    if hasattr(data, 'y_left_mult'):
        target['y_left_mult'] = data.y_left_mult.view(-1)
    if hasattr(data, 'y_right_mult'):
        target['y_right_mult'] = data.y_right_mult.view(-1)

    # Full mode adds reuse-aware labels. These are useful when the model must
    # decide whether an operand can be reused from existing graph nodes.
    if mode == 'full':
        # Six-class reuse pattern label. If unavailable, default to class 0,
        # which corresponds to no detected reuse pattern.
        if hasattr(data, 'reuse_pattern'):
            target['reuse_pattern'] = data.reuse_pattern.view(-1)
        else:
            target['reuse_pattern'] = torch.zeros_like(target['y_shift'])

        # Left operand reuse label and reused-node index.
        if hasattr(data, 'y_left'):
            target['y_left'] = data.y_left.view(-1)
        if hasattr(data, 'left_is_reuse'):
            target['left_is_reuse'] = data.left_is_reuse.view(-1)
        else:
            target['left_is_reuse'] = torch.zeros_like(target['y_shift'])

        # Right operand reuse label and reused-node index.
        if hasattr(data, 'y_right'):
            target['y_right'] = data.y_right.view(-1)
        if hasattr(data, 'right_is_reuse'):
            target['right_is_reuse'] = data.right_is_reuse.view(-1)
        else:
            target['right_is_reuse'] = torch.zeros_like(target['y_shift'])

    return target


def get_curr_mult_values(data):
    """
    Retrieve the current multiplier value being decomposed.

    Some dataset versions explicitly store `curr_mult_value`. If that field is
    absent, this function falls back to the raw target value.

    Args:
        data:
            PyTorch Geometric `Data` or `Batch` object.

    Returns:
        torch.Tensor:
            Current multiplier values with shape [B].
    """
    if hasattr(data, 'curr_mult_value'):
        return data.curr_mult_value.view(-1)
    else:
        return data.raw_target.view(-1)


# ============================================================
# Accuracy and Evaluation Metrics
# ============================================================

def compute_top1_accuracy(preds, labels, ignore_index=-1):
    """
    Compute standard top-1 classification accuracy.

    Args:
        preds (torch.Tensor):
            Predicted class indices with shape [B].
        labels (torch.Tensor):
            Ground-truth class indices with shape [B].
        ignore_index (int or None):
            Label value to ignore. Use None to evaluate all samples.

    Returns:
        float:
            Mean classification accuracy.
    """
    if ignore_index is not None:
        mask = labels != ignore_index
        if mask.sum() == 0:
            return 0.0
        preds  = preds[mask]
        labels = labels[mask]

    return (preds == labels).float().mean().item()


def compute_tolerance_accuracy(preds, labels, tolerance=3):
    """
    Compute accuracy under a symmetric absolute-error tolerance.

    This is useful for ordinal outputs such as shift values, where predictions
    close to the ground truth may still be informative.

    Args:
        preds (torch.Tensor): Predicted integer values.
        labels (torch.Tensor): Ground-truth integer values.
        tolerance (int): Maximum allowed absolute difference.

    Returns:
        float:
            Fraction of predictions satisfying |pred - label| <= tolerance.
    """
    errs = (preds - labels).abs()
    return float((errs <= tolerance).float().mean())


def compute_signed_tolerance_accuracy(preds, labels, tolerance, ignore_index=None):
    """
    Compute one-sided tolerance accuracy.

    Unlike symmetric tolerance accuracy, this metric evaluates whether the
    prediction falls within a directional interval. For example, a positive
    tolerance allows predictions in [label, label + tolerance], while a negative
    tolerance allows predictions in [label + tolerance, label].

    Args:
        preds (torch.Tensor): Predicted integer values.
        labels (torch.Tensor): Ground-truth integer values.
        tolerance (int): Signed tolerance range.
        ignore_index (int or None): Optional ignored label value.

    Returns:
        float:
            Directional tolerance accuracy.
    """
    if ignore_index is not None:
        mask = labels != ignore_index
        if mask.sum() == 0:
            return 0.0
        preds  = preds[mask]
        labels = labels[mask]

    if tolerance < 0:
        # Correct if prediction is within [label + tolerance, label].
        lower_bound = labels + tolerance
        upper_bound = labels
        return ((preds >= lower_bound) & (preds <= upper_bound)).float().mean().item()
    elif tolerance > 0:
        # Correct if prediction is within [label, label + tolerance].
        lower_bound = labels
        upper_bound = labels + tolerance
        return ((preds >= lower_bound) & (preds <= upper_bound)).float().mean().item()
    else:
        # Exact match when tolerance is zero.
        return (preds == labels).float().mean().item()


def compute_mult_accuracy(pred_mult_log, target_mult, tolerance=0.1):
    """
    Evaluate multiplier prediction using relative error in value space.

    The model predicts log2(mult + 1). This function converts predictions back
    to value space and checks whether the relative error is below a threshold.

    Args:
        pred_mult_log (torch.Tensor): Predicted log-space multiplier values.
        target_mult (torch.Tensor): Ground-truth multiplier values.
        tolerance (float): Relative-error threshold.

    Returns:
        float:
            Fraction of predictions with relative error below the threshold.
    """
    pred_mult_value = torch.pow(2, pred_mult_log) - 1.0
    pred_mult_value = pred_mult_value.clamp(min=1.0)

    target_mult_value = target_mult.float()
    rel_error = torch.abs(pred_mult_value - target_mult_value) / (target_mult_value + 1e-6)

    accuracy = (rel_error < tolerance).float().mean().item()
    return accuracy


def compute_mult_log_tolerance(pred_mult_log, target_mult, log_tolerance=1.0):
    """
    Evaluate multiplier prediction directly in log-space.

    This metric is more stable when multiplier values span several orders of
    magnitude, because an error of one log2 unit corresponds approximately to a
    factor-of-two difference.

    Args:
        pred_mult_log (torch.Tensor): Predicted log2(mult + 1) values.
        target_mult (torch.Tensor): Ground-truth multiplier values.
        log_tolerance (float): Allowed absolute error in log-space.

    Returns:
        float:
            Fraction of predictions within the log-space tolerance.
    """
    target_log = torch.log2(target_mult.float() + 1.0)
    log_diff   = torch.abs(pred_mult_log - target_log)
    accuracy   = (log_diff < log_tolerance).float().mean().item()
    return accuracy


def compute_reuse_decision_accuracy(reuse_pred, reuse_target):
    """
    Compute binary accuracy for operand reuse decisions.

    Args:
        reuse_pred (torch.Tensor):
            Predicted binary class indices after argmax, shape [B].
        reuse_target (torch.Tensor):
            Ground-truth reuse labels, either float or integer, shape [B].

    Returns:
        float:
            Binary reuse-decision accuracy.
    """
    if reuse_pred.ndim != 1:
        raise ValueError(f"reuse_pred must be 1D [B] (post-argmax), got shape {reuse_pred.shape}")
    if reuse_target.ndim != 1:
        raise ValueError(f"reuse_target must be 1D [B], got shape {reuse_target.shape}")

    # Convert floating-point labels to binary class labels.
    if reuse_target.dtype == torch.float32:
        reuse_target = (reuse_target > 0.5).long()
    else:
        reuse_target = reuse_target.long()

    accuracy = (reuse_pred == reuse_target).float().mean().item()
    return accuracy


def compute_reuse_node_accuracy(node_pred, node_target, node_mask):
    """
    Compute node-selection accuracy for reused operands.

    The metric is evaluated only on samples where reuse actually occurs. The
    node mask ensures that predictions are counted as correct only if they refer
    to a valid available graph node.

    Args:
        node_pred (torch.Tensor):
            Predicted node indices after argmax, shape [B].
        node_target (torch.Tensor):
            Ground-truth reused-node indices, shape [B]. Use -1 for no reuse.
        node_mask (torch.Tensor):
            Boolean mask of available nodes, shape [B, max_nodes].

    Returns:
        float:
            Accuracy over reuse-positive samples only.
    """
    if node_pred.ndim != 1:
        raise ValueError(f"node_pred must be 1D [B] (post-argmax), got shape {node_pred.shape}")
    if node_target.ndim != 1:
        raise ValueError(f"node_target must be 1D [B], got shape {node_target.shape}")
    if node_mask.ndim != 2:
        raise ValueError(f"node_mask must be 2D [B, max_nodes], got shape {node_mask.shape}")

    B      = node_pred.size(0)
    device = node_pred.device

    correct = torch.zeros(B, device=device, dtype=torch.bool)

    for i in range(B):
        if node_target[i] != -1:
            if node_target[i] < node_mask.size(1):
                correct[i] = (node_pred[i] == node_target[i]) and node_mask[i, node_target[i]]

    reuse_mask         = (node_target != -1)
    meaningful_samples = reuse_mask.sum().item()
    if meaningful_samples > 0:
        accuracy = correct[reuse_mask].float().mean().item()
    else:
        accuracy = 0.0

    return accuracy


# ============================================================
# Multi-Task Loss Function
# ============================================================

def compute_loss(pred, target, curr_mult_values=None, config=None, mode='full'):
    """
    Compute the weighted multi-task training loss.

    The loss combines several objectives:
        - shift prediction: multi-class cross entropy
        - operation prediction: multi-class cross entropy
        - left/right multiplier prediction: Smooth L1 loss in log-space
        - reuse pattern prediction: multi-class cross entropy
        - left/right reuse decision: binary cross entropy implemented as CE
        - left/right reuse node selection: masked multi-class cross entropy

    Args:
        pred (dict):
            Model prediction dictionary. Expected keys include `shift`, `op`,
            and optionally `left_mult`, `right_mult`, `reuse_pattern`,
            `left_reuse`, and `right_reuse`.
        target (dict):
            Target dictionary returned by `prepare_target_dict`.
        curr_mult_values:
            Reserved for future loss terms that depend on the current value
            being decomposed. Currently unused.
        config (dict or None):
            Optional loss configuration. If None, default task weights are used.
        mode (str):
            Use `'simple'` for core tasks only, or `'full'` for reuse-aware
            multi-task training.

    Returns:
        tuple:
            (total_loss, loss_dict), where total_loss is a differentiable tensor
            and loss_dict contains scalar diagnostics for logging.
    """
    device = pred["shift"].device

    # -------------------- Default task weights --------------------
    # The operation loss receives a relatively high weight because operation
    # prediction is the central classification task in the decomposition step.
    if config is None:
        if mode == 'full':
            weights = {
                'shift':                2.0,
                'op':                   10.0,
                'left_mult':            0.5,
                'right_mult':           0.5,
                'reuse_pattern':        2.0,
                'left_reuse_decision':  1.0,
                'right_reuse_decision': 1.0,
                'left_reuse_node':      0.5,
                'right_reuse_node':     0.5,
            }
        else:
            weights = {
                'shift':      2.0,
                'op':         10.0,
                'left_mult':  0.5,
                'right_mult': 0.5,
            }
    else:
        # Accept either a raw weight dictionary or a config object containing
        # a nested `weights` dictionary.
        weights = config if isinstance(config, dict) else config.get('weights', {})

    # -------------------- Shape validation --------------------
    # Explicit assertions make debugging easier when model heads or dataset
    # labels are changed during experimentation.
    try:
        assert pred["shift"].ndim == 2,     f"shift must be 2D, got {pred['shift'].shape}"
        assert pred["op"].ndim == 2,        f"op must be 2D, got {pred['op'].shape}"
        assert target["y_shift"].ndim == 1, f"y_shift must be 1D, got {target['y_shift'].shape}"
        assert target["y_op"].ndim == 1,    f"y_op must be 1D, got {target['y_op'].shape}"

        if mode == 'full':
            assert pred["reuse_pattern"].ndim == 2, f"reuse_pattern must be 2D, got {pred['reuse_pattern'].shape}"
            assert pred["left_reuse"]["reuse_decision"].ndim == 2
            assert pred["right_reuse"]["reuse_decision"].ndim == 2
            assert pred["left_reuse"]["node_scores"].ndim == 2
            assert pred["right_reuse"]["node_scores"].ndim == 2
    except AssertionError as e:
        print(f"\nDimension error: {e}")
        print("Prediction shapes:")
        for k, v in pred.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2.shape}")
            else:
                print(f"  {k}: {v.shape}")
        print("Target shapes:")
        for k, v in target.items():
            print(f"  {k}: {v.shape}")
        raise

    # -------------------- Core classification losses --------------------
    shift_loss = F.cross_entropy(pred["shift"], target["y_shift"], reduction='mean')
    op_loss = F.cross_entropy(pred["op"], target["y_op"], reduction='mean')

    # -------------------- Multiplier regression losses --------------------
    # Multipliers can be large, so regression is performed in log2(mult + 1)
    # space. Smooth L1 loss provides robustness to outliers.
    if "left_mult" in pred and "y_left_mult" in target:
        left_mult_pred   = pred["left_mult"].squeeze(-1)
        left_mult_target = torch.log2(target["y_left_mult"].float() + 1.0)
        left_mult_target = torch.clamp(left_mult_target, min=0.0, max=20.0)
        left_mult_loss   = F.smooth_l1_loss(
            left_mult_pred,
            left_mult_target,
            reduction='mean',
            beta=2.0
        )
    else:
        left_mult_loss = torch.tensor(0.0, device=device)

    if "right_mult" in pred and "y_right_mult" in target:
        right_mult_pred   = pred["right_mult"].squeeze(-1)
        right_mult_target = torch.log2(target["y_right_mult"].float() + 1.0)
        right_mult_target = torch.clamp(right_mult_target, min=0.0, max=20.0)
        right_mult_loss   = F.smooth_l1_loss(
            right_mult_pred,
            right_mult_target,
            reduction='mean',
            beta=2.0
        )
    else:
        right_mult_loss = torch.tensor(0.0, device=device)

    # -------------------- Reuse-aware losses --------------------
    if mode == 'full':
        # Predict the high-level reuse pattern class.
        if "reuse_pattern" in pred and "reuse_pattern" in target:
            reuse_pattern_loss = F.cross_entropy(
                pred["reuse_pattern"],
                target["reuse_pattern"].long(),
                reduction='mean'
            )
        else:
            reuse_pattern_loss = torch.tensor(0.0, device=device)

        # Predict whether the left operand reuses an existing graph node.
        if "left_reuse" in pred and "left_is_reuse" in target:
            left_reuse_decision_loss = F.cross_entropy(
                pred["left_reuse"]["reuse_decision"],
                target["left_is_reuse"].long() if target["left_is_reuse"].dtype != torch.long else target["left_is_reuse"],
                reduction='mean'
            )
        else:
            left_reuse_decision_loss = torch.tensor(0.0, device=device)

        # Predict whether the right operand reuses an existing graph node.
        if "right_reuse" in pred and "right_is_reuse" in target:
            right_reuse_decision_loss = F.cross_entropy(
                pred["right_reuse"]["reuse_decision"],
                target["right_is_reuse"].long() if target["right_is_reuse"].dtype != torch.long else target["right_is_reuse"],
                reduction='mean'
            )
        else:
            right_reuse_decision_loss = torch.tensor(0.0, device=device)

        # Node-selection losses are evaluated only when reuse is active.
        if "left_reuse" in pred and "y_left" in target and "left_is_reuse" in target:
            left_is_reuse_float = target["left_is_reuse"].float() if target["left_is_reuse"].dtype == torch.long else target["left_is_reuse"]
            left_reuse_mask = (left_is_reuse_float > 0.5) & (target["y_left"] != -1)

            if left_reuse_mask.any():
                left_reuse_node_loss = F.cross_entropy(
                    pred["left_reuse"]["node_scores"][left_reuse_mask],
                    target["y_left"][left_reuse_mask].long(),
                    reduction='mean'
                )
            else:
                left_reuse_node_loss = torch.tensor(0.0, device=device)
        else:
            left_reuse_node_loss = torch.tensor(0.0, device=device)

        if "right_reuse" in pred and "y_right" in target and "right_is_reuse" in target:
            right_is_reuse_float = target["right_is_reuse"].float() if target["right_is_reuse"].dtype == torch.long else target["right_is_reuse"]
            right_reuse_mask = (right_is_reuse_float > 0.5) & (target["y_right"] != -1)

            if right_reuse_mask.any():
                right_reuse_node_loss = F.cross_entropy(
                    pred["right_reuse"]["node_scores"][right_reuse_mask],
                    target["y_right"][right_reuse_mask].long(),
                    reduction='mean'
                )
            else:
                right_reuse_node_loss = torch.tensor(0.0, device=device)
        else:
            right_reuse_node_loss = torch.tensor(0.0, device=device)

    else:
        # In simple mode, reuse-related losses are disabled.
        reuse_pattern_loss        = torch.tensor(0.0, device=device)
        left_reuse_decision_loss  = torch.tensor(0.0, device=device)
        right_reuse_decision_loss = torch.tensor(0.0, device=device)
        left_reuse_node_loss      = torch.tensor(0.0, device=device)
        right_reuse_node_loss     = torch.tensor(0.0, device=device)

    # -------------------- Weighted loss aggregation --------------------
    if mode == 'full':
        total_loss = (
            weights.get('shift', 2.0)                * shift_loss +
            weights.get('op', 10.0)                  * op_loss +
            weights.get('left_mult', 0.5)            * left_mult_loss +
            weights.get('right_mult', 0.5)           * right_mult_loss +
            weights.get('reuse_pattern', 2.0)        * reuse_pattern_loss +
            weights.get('left_reuse_decision', 1.0)  * left_reuse_decision_loss +
            weights.get('right_reuse_decision', 1.0) * right_reuse_decision_loss +
            weights.get('left_reuse_node', 0.5)      * left_reuse_node_loss +
            weights.get('right_reuse_node', 0.5)     * right_reuse_node_loss
        )
    else:
        total_loss = (
            weights.get('shift', 2.0)      * shift_loss +
            weights.get('op', 10.0)        * op_loss +
            weights.get('left_mult', 0.5)  * left_mult_loss +
            weights.get('right_mult', 0.5) * right_mult_loss
        )

    # -------------------- Numerical safety checks --------------------
    # These guards prevent unstable batches from silently corrupting training.
    if not torch.isfinite(total_loss):
        print(f"WARNING: Non-finite total loss detected!")
        total_loss = torch.tensor(1000.0, device=device, requires_grad=True)

    if total_loss.item() > 100:
        print(f"WARNING: High loss detected: {total_loss.item():.2f}")
        print(f"  shift: {shift_loss.item():.4f},  op: {op_loss.item():.4f}")
        print(f"  left_mult: {left_mult_loss.item():.4f},  right_mult: {right_mult_loss.item():.4f}")
        if mode == 'full':
            print(f"  reuse_pattern:        {reuse_pattern_loss.item():.4f}")
            print(f"  left_reuse_decision:  {left_reuse_decision_loss.item():.4f}")
            print(f"  right_reuse_decision: {right_reuse_decision_loss.item():.4f}")
            print(f"  left_reuse_node:      {left_reuse_node_loss.item():.4f}")
            print(f"  right_reuse_node:     {right_reuse_node_loss.item():.4f}")

        total_loss = torch.clamp(total_loss, max=50.0)

    # Return scalar diagnostics for experiment logging.
    loss_dict = {
        'shift_loss':      shift_loss.item(),
        'op_loss':         op_loss.item(),
        'left_mult_loss':  left_mult_loss.item(),
        'right_mult_loss': right_mult_loss.item(),
    }

    if mode == 'full':
        loss_dict.update({
            'reuse_pattern_loss':        reuse_pattern_loss.item(),
            'left_reuse_decision_loss':  left_reuse_decision_loss.item(),
            'right_reuse_decision_loss': right_reuse_decision_loss.item(),
            'left_reuse_node_loss':      left_reuse_node_loss.item(),
            'right_reuse_node_loss':     right_reuse_node_loss.item(),
        })

    return total_loss, loss_dict


def get_default_loss_config(mode='full'):
    """
    Return the default task-weight configuration.

    Args:
        mode (str):
            `'full'` includes reuse-aware loss terms. Any other value returns
            the simple loss configuration.

    Returns:
        dict:
            Dictionary containing task weights under the `weights` key.
    """
    if mode == 'full':
        return {
            'weights': {
                'shift':                2.0,
                'op':                   10.0,
                'left_mult':            0.5,
                'right_mult':           0.5,
                'reuse_pattern':        2.0,
                'left_reuse_decision':  1.0,
                'right_reuse_decision': 1.0,
                'left_reuse_node':      0.5,
                'right_reuse_node':     0.5,
            }
        }
    else:
        return {
            'weights': {
                'shift':      2.0,
                'op':         10.0,
                'left_mult':  0.5,
                'right_mult': 0.5,
            }
        }


def print_loss_summary(loss_dict, mode='full'):
    """
    Print a formatted summary of loss components.

    This helper is intended for quick debugging and experiment monitoring during
    training. For formal experiments, these values should also be recorded by a
    logger such as TensorBoard, Weights & Biases, or a CSV/JSON log file.

    Args:
        loss_dict (dict):
            Dictionary returned by `compute_loss`.
        mode (str):
            Training mode used to determine whether reuse losses are printed.
    """
    print("\n" + "=" * 60)
    print("Loss Summary:")
    print("=" * 60)

    print(f"  Shift Loss:       {loss_dict.get('shift_loss', 0):.4f}")
    print(f"  Op Loss:          {loss_dict.get('op_loss', 0):.4f}")
    print(f"  Left Mult Loss:   {loss_dict.get('left_mult_loss', 0):.4f}")
    print(f"  Right Mult Loss:  {loss_dict.get('right_mult_loss', 0):.4f}")

    if mode == 'full':
        print(f"\n  Reuse Pattern Loss:       {loss_dict.get('reuse_pattern_loss', 0):.4f}")
        print(f"  Left Reuse Decision:      {loss_dict.get('left_reuse_decision_loss', 0):.4f}")
        print(f"  Right Reuse Decision:     {loss_dict.get('right_reuse_decision_loss', 0):.4f}")
        print(f"  Left Reuse Node:          {loss_dict.get('left_reuse_node_loss', 0):.4f}")
        print(f"  Right Reuse Node:         {loss_dict.get('right_reuse_node_loss', 0):.4f}")

    print("=" * 60 + "\n")
