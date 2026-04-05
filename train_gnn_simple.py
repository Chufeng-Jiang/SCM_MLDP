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
    set_seed
)

set_seed(seed=42)



def compute_simple_loss(pred, target, config):
    device = pred['op'].device
    
    # Op loss (交叉熵)
    op_loss = nn.functional.cross_entropy(
        pred['op'], 
        target['y_op'],
        reduction='mean'
    )
    
    total_loss = op_loss
    
    loss_dict = {
        'op_loss': op_loss.item()
    }
    
    return total_loss, loss_dict


def prepare_simple_target(data):
    batch_size = data.num_graphs
    device = data.x.device
    
    if hasattr(data, 'y_op'):
        y_op = data.y_op.view(batch_size)
    else:
        y_op = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    return {
        'y_op': y_op
    }


def compute_top1_accuracy(pred, target):
    if len(pred) == 0 or len(target) == 0:
        return 0.0
    correct = (pred == target).sum().item()
    total = len(target)
    return correct / total if total > 0 else 0.0


def train_one_epoch(model, loader, optimizer, device, config, scaler=None):
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
            data = data.to(device, non_blocking=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = model(data)
                    target = prepare_simple_target(data)
                    loss, loss_dict = compute_simple_loss(pred, target, config)
                    
                    if not torch.isfinite(loss):
                        print(f"\n⚠️ Warning: Invalid loss at batch {batch_idx}: {loss.item()}")
                        skipped_batches += 1
                        optimizer.zero_grad()
                        continue
                    
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                
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
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
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
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    
                    optimizer.step()
                    optimizer.zero_grad()

            batch_size = data.num_graphs
            total_loss += loss.item() * accumulation_steps * batch_size
            total_op_loss += loss_dict['op_loss'] * batch_size
            total_samples += batch_size
                
        except Exception as e:
            print(f"\n❌ Error at batch {batch_idx}: {e}")
            skipped_batches += 1
            optimizer.zero_grad()
            continue

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
    unique_sample_indices = list(set(idx for idx, k in dataset.index_map))
    
    train_sample_indices, val_sample_indices = train_test_split(
        unique_sample_indices, 
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True
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
    
    print(f"\n{'='*80}")
    print(f"Training SimpleSCMGNN - Op Prediction Only")
    print(f"LR Strategy: {config['lr_scheduler']}")
    print(f"Device: {device}")
    print(f"{'='*80}")
    
    for epoch in range(1, config['epochs'] + 1):
        try:
            train_loss, train_components = train_one_epoch(
                model, train_loader, optimizer, device, config, scaler
            )
            
            train_metrics = evaluate(model, train_loader, device, config, use_amp=False)
            val_metrics = evaluate(model, val_loader, device, config, use_amp=False)
            
            current_lr = get_current_lr(optimizer)
            
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
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                wait = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'val_metrics': val_metrics,
                    'config': config
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
            print(f"❌ Error at epoch {epoch}: {e}")
            print("Continuing with next epoch...")
            continue
    
    print(f"\n{'='*60}")
    print(f"Final Evaluation on Test Set")
    print(f"{'='*60}")
    
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
    
    history_df = pd.DataFrame(train_history)
    os.makedirs("./training_history", exist_ok=True)
    history_df.to_csv("./training_history/training_history_simple.csv", index=False)

    test_results = {
        'final_epoch': len(train_history),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_lr': current_lr,
        **test_metrics
    }
    
    test_results_df = pd.DataFrame([test_results])
    test_results_df.to_csv("./model_results/test_results_simple.csv", index=False)
    
    print(f"\n✅ Training history saved: training_history/training_history_simple.csv")
    print(f"✅ Test results saved: model_results/test_results_simple.csv")
    print(f"✅ Best model: epoch {best_epoch}, val_loss: {best_val_loss:.4f}")
    
    return test_results, best_model_state



def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    
    config = {
        'batch_size': 256,
        'accumulation_steps': 1,
        'epochs': 800,
        'patience': 20,
        'print_every': 10,
        'use_amp': False,
        'node_in_dim': 199,  
        'edge_in_dim': 12,
        'hidden_dim': 128,
        'num_heads': 8,
        'dropout': 0.1,
        'conv_type': 'gatv2',
        'num_gnn_layers': 5,
        'val_ratio': 0.1,
        'random_state': 42,
        'lr': 2e-4,
        'weight_decay': 1e-3,
        'grad_clip': 5.0,
        

        'lr_scheduler': 'cosine_warm_restarts',
        'min_lr': 1e-6,
        'restart_t0': 60,
        'restart_t_mult': 2,
        'warmup_steps': 100,
        'warmup_start_factor': 0.1,
    }
    
    os.makedirs("./model_results", exist_ok=True)
    os.makedirs("./training_history", exist_ok=True)
    
    print(f"\n📋 Model Configuration (Simple - OP Only):")
    print(f"  Node Input Dim: {config['node_in_dim']}")
    print(f"  Hidden Dim: {config['hidden_dim']}")
    print(f"  GNN Layers: {config['num_gnn_layers']}")
    print(f"  Conv Type: {config['conv_type']}")
    print(f"  Task: Op prediction only (4 classes)")
    
    print("\n📦 Loading dataset splits...")
    from data_split import load_split_targets
    train_targets, test_targets = load_split_targets()
    
    print("\n📦 Loading training dataset...")
    train_dataset = SCMGraphDataset(
        "./data/split/train_data.json",
        max_prefix_len=11,
        split_type='train',
        train_targets=train_targets,
        test_targets=test_targets
    )
    print(f"✅ Training dataset: {len(train_dataset.samples)} samples → {len(train_dataset)} graphs")
    
    print("\n📦 Loading test dataset...")
    test_dataset = SCMGraphDataset(
        "./data/split/test_data.json",
        max_prefix_len=11,
        split_type='test',
        train_targets=train_targets,
        test_targets=test_targets
    )
    print(f"✅ Test dataset: {len(test_dataset.samples)} samples → {len(test_dataset)} graphs")
    
    train_indices, val_indices = create_data_splits_by_sample(
        train_dataset, 
        val_ratio=config['val_ratio'],
        random_state=config['random_state']
    )
    
    print(f"🔧 Creating optimized data loaders with {get_optimal_workers()} workers...")
    train_loader = create_optimized_dataloader(
        Subset(train_dataset, train_indices), 
        batch_size=config['batch_size'], 
        shuffle=True,
        is_train=True
    )
    val_loader = create_optimized_dataloader(
        Subset(train_dataset, val_indices), 
        batch_size=config['batch_size'], 
        shuffle=False,
        is_train=False
    )
    test_loader = create_optimized_dataloader(
        test_dataset,  
        batch_size=config['batch_size'], 
        shuffle=False,
        is_train=False
    )
    
    print(f"\n📊 Final Data Loaders (Graphs):")
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