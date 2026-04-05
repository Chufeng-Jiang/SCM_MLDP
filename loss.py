import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def prepare_target_dict(data, mode='full'):
    target = {
        'y_shift': data.y_shift.view(-1),
        'y_op':    data.y_op.view(-1),
    }

    # Mult regression tasks
    if hasattr(data, 'y_left_mult'):
        target['y_left_mult'] = data.y_left_mult.view(-1)
    if hasattr(data, 'y_right_mult'):
        target['y_right_mult'] = data.y_right_mult.view(-1)

    # Full mode: add reuse-related labels
    if mode == 'full':
        # Reuse pattern label (6-class)
        if hasattr(data, 'reuse_pattern'):
            target['reuse_pattern'] = data.reuse_pattern.view(-1)
        else:
            target['reuse_pattern'] = torch.zeros_like(target['y_shift'])

        # Left operand reuse decision and position
        if hasattr(data, 'y_left'):
            target['y_left'] = data.y_left.view(-1)
        if hasattr(data, 'left_is_reuse'):
            target['left_is_reuse'] = data.left_is_reuse.view(-1)
        else:
            target['left_is_reuse'] = torch.zeros_like(target['y_shift'])

        # Right operand reuse decision and position
        if hasattr(data, 'y_right'):
            target['y_right'] = data.y_right.view(-1)
        if hasattr(data, 'right_is_reuse'):
            target['right_is_reuse'] = data.right_is_reuse.view(-1)
        else:
            target['right_is_reuse'] = torch.zeros_like(target['y_shift'])

    return target


def get_curr_mult_values(data):
    """Get the current mult value to be decomposed."""
    if hasattr(data, 'curr_mult_value'):
        return data.curr_mult_value.view(-1)
    else:
        return data.raw_target.view(-1)


# ==================== Accuracy Functions ====================

def compute_top1_accuracy(preds, labels, ignore_index=-1):
    if ignore_index is not None:
        mask = labels != ignore_index
        if mask.sum() == 0:
            return 0.0
        preds  = preds[mask]
        labels = labels[mask]

    return (preds == labels).float().mean().item()


def compute_tolerance_accuracy(preds, labels, tolerance=3):
    errs = (preds - labels).abs()
    return float((errs <= tolerance).float().mean())


def compute_signed_tolerance_accuracy(preds, labels, tolerance, ignore_index=None):
    if ignore_index is not None:
        mask = labels != ignore_index
        if mask.sum() == 0:
            return 0.0
        preds  = preds[mask]
        labels = labels[mask]

    if tolerance < 0:
        # Correct if prediction is in [labels+tolerance, labels]
        lower_bound = labels + tolerance
        upper_bound = labels
        return ((preds >= lower_bound) & (preds <= upper_bound)).float().mean().item()
    elif tolerance > 0:
        # Correct if prediction is in [labels, labels+tolerance]
        lower_bound = labels
        upper_bound = labels + tolerance
        return ((preds >= lower_bound) & (preds <= upper_bound)).float().mean().item()
    else:
        # tolerance == 0: exact match required
        return (preds == labels).float().mean().item()


def compute_mult_accuracy(pred_mult_log, target_mult, tolerance=0.1):
    pred_mult_value = torch.pow(2, pred_mult_log) - 1.0  
    pred_mult_value = pred_mult_value.clamp(min=1.0)

    target_mult_value = target_mult.float()  
    rel_error = torch.abs(pred_mult_value - target_mult_value) / (target_mult_value + 1e-6)

    accuracy = (rel_error < tolerance).float().mean().item()
    return accuracy


def compute_mult_log_tolerance(pred_mult_log, target_mult, log_tolerance=1.0):
    target_log = torch.log2(target_mult.float() + 1.0)
    log_diff   = torch.abs(pred_mult_log - target_log)
    accuracy   = (log_diff < log_tolerance).float().mean().item()
    return accuracy


def compute_reuse_decision_accuracy(reuse_pred, reuse_target):

    if reuse_pred.ndim != 1:
        raise ValueError(f"reuse_pred must be 1D [B] (post-argmax), got shape {reuse_pred.shape}")
    if reuse_target.ndim != 1:
        raise ValueError(f"reuse_target must be 1D [B], got shape {reuse_target.shape}")

    # Convert float target to binary (>0.5 -> 1)
    if reuse_target.dtype == torch.float32:
        reuse_target = (reuse_target > 0.5).long()
    else:
        reuse_target = reuse_target.long()

    accuracy = (reuse_pred == reuse_target).float().mean().item()
    return accuracy


def compute_reuse_node_accuracy(node_pred, node_target, node_mask):
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
        if node_target[i] != -1:  # reuse operation
            if node_target[i] < node_mask.size(1):
                correct[i] = (node_pred[i] == node_target[i]) and node_mask[i, node_target[i]]

    reuse_mask         = (node_target != -1)
    meaningful_samples = reuse_mask.sum().item()
    if meaningful_samples > 0:
        accuracy = correct[reuse_mask].float().mean().item()
    else:
        accuracy = 0.0

    return accuracy


# ==================== Loss Functions ====================

def compute_loss(pred, target, curr_mult_values=None, config=None, mode='full'):
    device = pred["shift"].device

    # ==================== Default Weight Config ====================
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
        else:  # simple mode
            weights = {
                'shift':      2.0,
                'op':         10.0,
                'left_mult':  0.5,
                'right_mult': 0.5,
            }
    else:
        weights = config if isinstance(config, dict) else config.get('weights', {})

    # ==================== Dimension Validation ====================
    try:
        assert pred["shift"].ndim == 2,    f"shift must be 2D, got {pred['shift'].shape}"
        assert pred["op"].ndim == 2,       f"op must be 2D, got {pred['op'].shape}"
        assert target["y_shift"].ndim == 1, f"y_shift must be 1D, got {target['y_shift'].shape}"
        assert target["y_op"].ndim == 1,   f"y_op must be 1D, got {target['y_op'].shape}"

        if mode == 'full':
            assert pred["reuse_pattern"].ndim == 2,              f"reuse_pattern must be 2D, got {pred['reuse_pattern'].shape}"
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

    shift_loss = F.cross_entropy(pred["shift"], target["y_shift"], reduction='mean')
    op_loss = F.cross_entropy(pred["op"], target["y_op"], reduction='mean')
    
    if "left_mult" in pred and "y_left_mult" in target:
        left_mult_pred   = pred["left_mult"].squeeze(-1)                          # [B]
        left_mult_target = torch.log2(target["y_left_mult"].float() + 1.0)       # [B]
        left_mult_target = torch.clamp(left_mult_target, min=0.0, max=20.0)
        left_mult_loss   = F.smooth_l1_loss(left_mult_pred, left_mult_target, reduction='mean', beta=2.0)
    else:
        left_mult_loss = torch.tensor(0.0, device=device)

    if "right_mult" in pred and "y_right_mult" in target:
        right_mult_pred   = pred["right_mult"].squeeze(-1)                        # [B]
        right_mult_target = torch.log2(target["y_right_mult"].float() + 1.0)     # [B]
        right_mult_target = torch.clamp(right_mult_target, min=0.0, max=20.0)
        right_mult_loss   = F.smooth_l1_loss(right_mult_pred, right_mult_target, reduction='mean', beta=2.0)
    else:
        right_mult_loss = torch.tensor(0.0, device=device)


    if mode == 'full':
        if "reuse_pattern" in pred and "reuse_pattern" in target:
            reuse_pattern_loss = F.cross_entropy(
                pred["reuse_pattern"],
                target["reuse_pattern"].long(),
                reduction='mean'
            )
        else:
            reuse_pattern_loss = torch.tensor(0.0, device=device)

        if "left_reuse" in pred and "left_is_reuse" in target:
            left_reuse_decision_loss = F.cross_entropy(
                pred["left_reuse"]["reuse_decision"],
                target["left_is_reuse"].long() if target["left_is_reuse"].dtype != torch.long else target["left_is_reuse"],
                reduction='mean'
            )
        else:
            left_reuse_decision_loss = torch.tensor(0.0, device=device)

        if "right_reuse" in pred and "right_is_reuse" in target:
            right_reuse_decision_loss = F.cross_entropy(
                pred["right_reuse"]["reuse_decision"],
                target["right_is_reuse"].long() if target["right_is_reuse"].dtype != torch.long else target["right_is_reuse"],
                reduction='mean'
            )
        else:
            right_reuse_decision_loss = torch.tensor(0.0, device=device)

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
        reuse_pattern_loss        = torch.tensor(0.0, device=device)
        left_reuse_decision_loss  = torch.tensor(0.0, device=device)
        right_reuse_decision_loss = torch.tensor(0.0, device=device)
        left_reuse_node_loss      = torch.tensor(0.0, device=device)
        right_reuse_node_loss     = torch.tensor(0.0, device=device)

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

    print("\n" + "="*60)
    print("Loss Summary:")
    print("="*60)

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

    print("="*60 + "\n")