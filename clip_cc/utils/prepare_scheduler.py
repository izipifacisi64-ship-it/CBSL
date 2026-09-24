# clip_cc/utils/prepare_scheduler.py
# -*- coding: utf-8 -*-
"""
线性衰减学习率调度器
═══════════════════════════════════════════
从 3.5e-6 线性衰减到 0

调度策略：
  epoch=0       → 3.5e-6  (起点)
  epoch=T-1     → 0       (终点)
  中间线性插值
"""

import math


class LinearDecayLR:
    """
    线性衰减学习率调度器。
    从 base_lr 线性衰减到 min_lr=0。

    阶段
    ────
    epoch 0 ~ total_epochs-1
    从 base_lr 线性衰减到 0
    """

    def __init__(self, optimizer, total_epochs=30, min_lr=0.0):
        self.optimizer     = optimizer
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.last_epoch    = -1
        self.base_lrs      = [g['lr'] for g in optimizer.param_groups]
        self.step()   # 初始化，对齐 PyTorch 惯例

    def _scale(self, epoch):
        T = self.total_epochs
        
        if T <= 1:
            return 1.0
        
        # 线性衰减: epoch=0 → 1.0, epoch=T-1 → min_lr/base_lr
        progress = epoch / (T - 1)
        # 线性插值：从 base_lr 到 min_lr
        return 1.0 - (1.0 - self.min_lr / self.base_lrs[0]) * progress

    def step(self):
        self.last_epoch += 1
        scale = self._scale(self.last_epoch)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group['lr'] = base_lr * scale

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items()
                if k != 'optimizer'}

    def load_state_dict(self, state):
        self.__dict__.update(state)


def create_scheduler(optimizer, args=None):
    scheduler_type = 'linear'
    print("Using linear decay scheduler type")

    total_epochs = getattr(args, 'epochs', 30) if args else 30

    if scheduler_type == 'linear':
        # 设置初始学习率为 3.5e-6
        for group in optimizer.param_groups:
            group['lr'] = 3.5e-6
        
        lr_scheduler = LinearDecayLR(
            optimizer,
            total_epochs=total_epochs,
            min_lr=0.0,           # 终点为 0
        )

        base_lr = lr_scheduler.base_lrs[0]
        T = lr_scheduler.total_epochs
        print(f'[Scheduler] LinearDecayLR')
        print(f'  linear decay : epoch 0~{T-1}  {base_lr:.2e} → {lr_scheduler.min_lr:.2e}')

    else:
        raise ValueError(f'Invalid scheduler type {scheduler_type}!')

    return lr_scheduler