"""
损失函数
包括上下文学习损失和任务特定损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
import numpy as np

from config.model_config import KumoRFMConfig


class ICLLoss(nn.Module):
    """
    上下文学习损失
    根据任务类型动态选择损失函数
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config

        # 任务特定损失
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.ranking_loss = ListwiseLoss()

        # 辅助损失权重
        self.auxiliary_loss_weight = 0.1

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: torch.Tensor,
                task_type: str,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算损失

        predictions: 模型预测字典
        targets: 真实标签
        task_type: 任务类型
        mask: 有效样本掩码
        """
        # 主损失
        if task_type in ['classification', 'exists']:
            loss = self._classification_loss(predictions, targets, mask)

        elif task_type in ['regression', 'count', 'sum']:
            loss = self._regression_loss(predictions, targets, mask)

        elif task_type in ['link_prediction', 'next']:
            loss = self._ranking_loss(predictions, targets, mask)

        else:
            raise ValueError(f"未知的任务类型: {task_type}")

        # 添加辅助损失（如果有）
        if 'auxiliary_loss' in predictions:
            loss += self.auxiliary_loss_weight * predictions['auxiliary_loss']

        return loss

    def _classification_loss(self,
                             predictions: Dict[str, torch.Tensor],
                             targets: torch.Tensor,
                             mask: Optional[torch.Tensor]) -> torch.Tensor:
        """分类损失"""
        logits = predictions['logits']

        # 处理掩码
        if mask is not None:
            logits = logits[mask]
            targets = targets[mask]

        # 确保目标是长整型
        targets = targets.long()

        # 计算交叉熵
        loss = self.classification_loss(logits, targets)

        return loss

    def _regression_loss(self,
                         predictions: Dict[str, torch.Tensor],
                         targets: torch.Tensor,
                         mask: Optional[torch.Tensor]) -> torch.Tensor:
        """回归损失"""
        preds = predictions['predictions']

        # 处理掩码
        if mask is not None:
            preds = preds[mask]
            targets = targets[mask]

        # 计算MSE
        loss = self.regression_loss(preds, targets)

        # 可选：添加Huber损失以提高鲁棒性
        # loss = F.smooth_l1_loss(preds, targets)

        return loss

    def _ranking_loss(self,
                      predictions: Dict[str, torch.Tensor],
                      targets: torch.Tensor,
                      mask: Optional[torch.Tensor]) -> torch.Tensor:
        """排序损失"""
        scores = predictions.get('scores', predictions.get('logits'))

        # 使用自定义的列表损失
        loss = self.ranking_loss(scores, targets, mask)

        return loss


class ListwiseLoss(nn.Module):
    """
    列表级排序损失
    用于推荐等排序任务
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        scores: (batch_size, num_items) 预测分数
        targets: (batch_size, num_items) 真实相关性或二进制标签
        mask: (batch_size, num_items) 有效项掩码
        """
        # 应用温度缩放
        scores = scores / self.temperature

        # 计算softmax
        if mask is not None:
            # 掩码无效位置
            scores = scores.masked_fill(~mask, float('-inf'))

        probs = F.softmax(scores, dim=-1)

        # 归一化目标
        if mask is not None:
            targets = targets * mask.float()
            target_sum = targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            target_sum = targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        targets_normalized = targets / target_sum

        # 计算KL散度
        loss = F.kl_div(
            probs.log(),
            targets_normalized,
            reduction='batchmean'
        )

        return loss


class ContrastiveLoss(nn.Module):
    """
    对比学习损失
    用于学习更好的表示
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor,
                positive: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        """
        anchor: (batch_size, hidden_dim) 锚点嵌入
        positive: (batch_size, hidden_dim) 正样本嵌入
        negatives: (batch_size, num_negatives, hidden_dim) 负样本嵌入
        """
        batch_size = anchor.shape[0]

        # 归一化
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-2)

        # 计算相似度
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        neg_sim = torch.matmul(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / self.temperature

        # 拼接所有相似度
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        # 标签（第一个是正样本）
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        # 计算交叉熵
        loss = F.cross_entropy(logits, labels)

        return loss


class ConsistencyLoss(nn.Module):
    """
    一致性损失
    确保相似输入产生相似输出
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred1: torch.Tensor, pred2: torch.Tensor,
                similarity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        pred1, pred2: 两个预测
        similarity: 输入相似度（可选）
        """
        # 计算预测差异
        diff = F.mse_loss(pred1, pred2, reduction='none')

        if similarity is not None:
            # 根据输入相似度加权
            loss = (diff * similarity).mean()
        else:
            loss = diff.mean()

        return loss


class MultiTaskLoss(nn.Module):
    """
    多任务学习损失
    动态平衡不同任务的损失
    """

    def __init__(self, num_tasks: int = 4):
        super().__init__()
        self.num_tasks = num_tasks

        # 可学习的任务权重
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        losses: {任务名: 损失值}
        """
        total_loss = 0

        for i, (task_name, loss) in enumerate(losses.items()):
            # 使用不确定性加权
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]

        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    处理类别不平衡问题
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch_size, num_classes)
        targets: (batch_size,)
        """
        # 计算概率
        probs = F.softmax(logits, dim=-1)

        # 获取目标类别的概率
        batch_size = logits.shape[0]
        class_probs = probs[torch.arange(batch_size), targets]

        # 计算focal权重
        focal_weight = (1 - class_probs) ** self.gamma

        # 计算加权交叉熵
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = self.alpha * focal_weight * ce_loss

        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失
    提高模型泛化能力
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch_size, num_classes)
        targets: (batch_size,)
        """
        # 创建平滑标签
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        # 计算KL散度
        loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            true_dist,
            reduction='batchmean'
        )

        return loss


def get_loss_function(config: KumoRFMConfig, task_type: str) -> nn.Module:
    """
    根据任务类型获取损失函数
    """
    if task_type == 'pretrain':
        return ICLLoss(config)

    elif task_type == 'classification':
        return nn.CrossEntropyLoss()

    elif task_type == 'regression':
        return nn.MSELoss()

    elif task_type == 'ranking':
        return ListwiseLoss()

    elif task_type == 'contrastive':
        return ContrastiveLoss()

    else:
        raise ValueError(f"未知的任务类型: {task_type}")