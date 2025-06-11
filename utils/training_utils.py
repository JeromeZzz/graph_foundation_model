"""
训练工具函数
包括早停、模型检查点、学习率调度等
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
import json
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    早停机制
    监控验证集性能，在性能不再提升时停止训练
    """

    def __init__(self, patience: int = 10, delta: float = 0.0, mode: str = 'min'):
        """
        patience: 容忍的epoch数
        delta: 最小改进阈值
        mode: 'min' 或 'max'
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停

        score: 当前验证分数
        返回: 是否触发早停
        """
        if self.mode == 'min':
            is_better = score < self.best_score - self.delta
        else:
            is_better = score > self.best_score + self.delta

        if is_better:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f'早停计数器: {self.counter}/{self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
                logger.info('触发早停！')

        return self.early_stop

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.early_stop = False

        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')


class ModelCheckpoint:
    """
    模型检查点
    保存最佳模型和训练状态
    """

    def __init__(self,
                 save_dir: str = 'checkpoints',
                 filename_prefix: str = 'model',
                 save_best_only: bool = True,
                 mode: str = 'min',
                 save_freq: int = 1):
        """
        save_dir: 保存目录
        filename_prefix: 文件名前缀
        save_best_only: 是否只保存最佳模型
        mode: 'min' 或 'max'
        save_freq: 保存频率（epoch）
        """
        self.save_dir = save_dir
        self.filename_prefix = filename_prefix
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 最佳分数
        if mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

        self.best_model_path = None

    def save(self,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             epoch: int,
             score: float,
             val_score: Optional[float] = None,
             additional_info: Optional[Dict[str, Any]] = None):
        """
        保存检查点

        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        score: 训练分数
        val_score: 验证分数（用于决定是否保存）
        additional_info: 额外信息
        """
        # 检查是否应该保存
        should_save = False

        if not self.save_best_only and epoch % self.save_freq == 0:
            should_save = True

        # 检查是否是最佳模型
        check_score = val_score if val_score is not None else score

        if self.mode == 'min':
            is_best = check_score < self.best_score
        else:
            is_best = check_score > self.best_score

        if is_best:
            self.best_score = check_score
            should_save = True

            # 删除之前的最佳模型
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)

        if not should_save and self.save_best_only:
            return

        # 构建文件名
        if is_best and self.save_best_only:
            filename = f'{self.filename_prefix}_best.pt'
        else:
            filename = f'{self.filename_prefix}_epoch_{epoch}.pt'

        filepath = os.path.join(self.save_dir, filename)

        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_score': score,
            'val_score': val_score,
            'best_score': self.best_score,
            'timestamp': datetime.now().isoformat()
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, filepath)
        logger.info(f'保存检查点到: {filepath}')

        if is_best:
            self.best_model_path = filepath
            logger.info(f'新的最佳模型！分数: {check_score:.4f}')

        # 保存检查点信息
        self._save_checkpoint_info(checkpoint, filepath)

    def load(self,
             model: torch.nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载检查点

        model: 模型
        optimizer: 优化器（可选）
        checkpoint_path: 检查点路径（默认加载最佳模型）

        返回: 检查点字典
        """
        if checkpoint_path is None:
            if self.best_model_path:
                checkpoint_path = self.best_model_path
            else:
                # 找到最新的检查点
                checkpoints = [f for f in os.listdir(self.save_dir)
                               if f.startswith(self.filename_prefix) and f.endswith('.pt')]
                if not checkpoints:
                    raise ValueError(f"在 {self.save_dir} 中没有找到检查点")

                checkpoints.sort()
                checkpoint_path = os.path.join(self.save_dir, checkpoints[-1])

        logger.info(f'加载检查点: {checkpoint_path}')

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    def _save_checkpoint_info(self, checkpoint: Dict[str, Any], filepath: str):
        """保存检查点信息到JSON文件"""
        info = {
            'filepath': filepath,
            'epoch': checkpoint['epoch'],
            'train_score': checkpoint.get('train_score'),
            'val_score': checkpoint.get('val_score'),
            'timestamp': checkpoint.get('timestamp')
        }

        info_path = os.path.join(self.save_dir, 'checkpoint_info.json')

        # 读取现有信息
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                all_info = json.load(f)
        else:
            all_info = []

        # 添加新信息
        all_info.append(info)

        # 保存
        with open(info_path, 'w') as f:
            json.dump(all_info, f, indent=2)


class GradientAccumulator:
    """
    梯度累积器
    用于在小批量上模拟大批量训练
    """

    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def should_step(self) -> bool:
        """是否应该执行优化器步骤"""
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0

    def reset(self):
        """重置计数器"""
        self.step_count = 0


class MetricTracker:
    """
    指标跟踪器
    跟踪训练过程中的各种指标
    """

    def __init__(self, metrics: List[str], phases: List[str] = ['train', 'val']):
        """
        metrics: 要跟踪的指标名称
        phases: 训练阶段
        """
        self.metrics = metrics
        self.phases = phases

        # 初始化历史记录
        self.history = {
            phase: {metric: [] for metric in metrics}
            for phase in phases
        }

        # 当前epoch的值
        self.current = {
            phase: {metric: 0.0 for metric in metrics}
            for phase in phases
        }

        # 累积值（用于计算平均）
        self.accumulation = {
            phase: {metric: 0.0 for metric in metrics}
            for phase in phases
        }

        # 计数器
        self.counts = {phase: 0 for phase in phases}

    def update(self, phase: str, metrics: Dict[str, float], n: int = 1):
        """
        更新指标

        phase: 训练阶段
        metrics: 指标字典
        n: 样本数
        """
        for metric, value in metrics.items():
            if metric in self.metrics:
                self.accumulation[phase][metric] += value * n

        self.counts[phase] += n

    def compute_average(self, phase: str) -> Dict[str, float]:
        """计算平均值"""
        averages = {}

        if self.counts[phase] > 0:
            for metric in self.metrics:
                averages[metric] = self.accumulation[phase][metric] / self.counts[phase]
                self.current[phase][metric] = averages[metric]

        return averages

    def epoch_end(self):
        """epoch结束时调用"""
        # 保存当前epoch的平均值到历史
        for phase in self.phases:
            if self.counts[phase] > 0:
                averages = self.compute_average(phase)
                for metric in self.metrics:
                    self.history[phase][metric].append(averages[metric])

        # 重置累积值和计数器
        self.reset_epoch()

    def reset_epoch(self):
        """重置epoch状态"""
        for phase in self.phases:
            for metric in self.metrics:
                self.accumulation[phase][metric] = 0.0
            self.counts[phase] = 0

    def get_best(self, phase: str, metric: str, mode: str = 'max') -> Tuple[float, int]:
        """
        获取最佳值和对应的epoch

        返回: (最佳值, epoch索引)
        """
        values = self.history[phase][metric]
        if not values:
            return None, None

        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)

        return values[best_idx], best_idx

    def save(self, filepath: str):
        """保存历史到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load(self, filepath: str):
        """从文件加载历史"""
        with open(filepath, 'r') as f:
            self.history = json.load(f)


def set_random_seed(seed: int = 42):
    """设置随机种子以确保可重现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python内置随机
    import random
    random.seed(seed)

    logger.info(f"设置随机种子: {seed}")


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"总参数数: {total_params:,}")
    logger.info(f"可训练参数数: {trainable_params:,}")

    return trainable_params


def save_config(config: Any, filepath: str):
    """保存配置到文件"""
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config

    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"配置已保存到: {filepath}")


def create_optimizer(model: torch.nn.Module,
                     optimizer_name: str,
                     learning_rate: float,
                     weight_decay: float = 0.01,
                     **kwargs) -> torch.optim.Optimizer:
    """
    创建优化器

    optimizer_name: 优化器名称
    learning_rate: 学习率
    weight_decay: 权重衰减
    """
    # 分离需要和不需要权重衰减的参数
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'bias' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    # 创建优化器
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=learning_rate, **kwargs)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, **kwargs)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=learning_rate, momentum=0.9, **kwargs)
    else:
        raise ValueError(f"未知的优化器: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer,
                     scheduler_name: str,
                     num_epochs: int,
                     **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """
    创建学习率调度器
    """
    if scheduler_name.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, **kwargs
        )
    elif scheduler_name.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=num_epochs // 3, gamma=0.1, **kwargs
        )
    elif scheduler_name.lower() == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95, **kwargs
        )
    elif scheduler_name.lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, **kwargs
        )
    else:
        raise ValueError(f"未知的调度器: {scheduler_name}")

    return scheduler