"""utils.py"""
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
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# ==================== 优化器和调度器 ====================

def create_optimizer_and_scheduler(model, config):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 1. 定义 warmup（线性 warmup 到 lr）
    if config.get('warmup_steps', 0) > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=config.get('warmup_start_factor', 0.01),  # 起始 1% 学习率
            total_iters=config['warmup_steps']
        )
    else:
        warmup_scheduler = None

    # 2. 定义 main scheduler
    if config['lr_scheduler'] == 'cosine':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['cosine_t_max'],
            eta_min=config['min_lr']
        )
    elif config['lr_scheduler'] == 'cosine_warm_restarts':
        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['restart_t0'],
            T_mult=config['restart_t_mult'],
            eta_min=config['min_lr']
        )
    elif config['lr_scheduler'] == 'step':
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['step_gamma']
        )
    else:
        main_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    # 3. 合并 warmup + main scheduler
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
    """获取当前学习率"""
    return optimizer.param_groups[0]['lr']

def get_optimal_workers():
    """根据CPU核心数返回最优worker数"""
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 4  # 默认值
    # 经验公式: 留2个核心给系统和其他任务
    optimal = max(2, min(8, cpu_count - 2))
    return optimal

def create_optimized_dataloader(dataset, batch_size, shuffle=True, is_train=True):
    """创建优化的数据加载器"""
    num_workers = get_optimal_workers() if is_train else 2
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # 加速GPU传输
        persistent_workers=num_workers > 0,  # 保持worker进程
        prefetch_factor=2 if is_train else 1,  # 预取batch数
        drop_last=True,
    )
    return loader