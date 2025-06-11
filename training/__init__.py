"""
训练模块
包含预训练、微调和损失函数
"""

from .pretrain import PreTrainer, PretrainDataset
from .finetune import FineTuner, FinetuneDataset
from .loss import (
    ICLLoss,
    ListwiseLoss,
    ContrastiveLoss,
    ConsistencyLoss,
    MultiTaskLoss,
    FocalLoss,
    LabelSmoothingLoss,
    get_loss_function
)

__all__ = [
    # 训练器
    'PreTrainer',
    'FineTuner',

    # 数据集
    'PretrainDataset',
    'FinetuneDataset',

    # 损失函数
    'ICLLoss',
    'ListwiseLoss',
    'ContrastiveLoss',
    'ConsistencyLoss',
    'MultiTaskLoss',
    'FocalLoss',
    'LabelSmoothingLoss',
    'get_loss_function'
]

# 训练配置默认值
DEFAULT_TRAINING_CONFIG = {
    'learning_rate': 1e-4,
    'batch_size': 32,
    'num_epochs': 100,
    'warmup_steps': 1000,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'early_stopping_patience': 10,
    'save_frequency': 1
}

# 支持的优化器
SUPPORTED_OPTIMIZERS = [
    'adam',
    'adamw',
    'sgd',
    'rmsprop',
    'adagrad'
]

# 支持的学习率调度器
SUPPORTED_SCHEDULERS = [
    'cosine',
    'linear',
    'polynomial',
    'exponential',
    'step',
    'plateau'
]