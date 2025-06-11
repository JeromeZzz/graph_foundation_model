"""
上下文学习（In-Context Learning）模块
实现KumoRFM的核心ICL功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math

from config.model_config import KumoRFMConfig
from ..transformers.attention import CrossAttention, SelfAttentionBlock


class InContextLearningModule(nn.Module):
    """
    上下文学习模块
    处理上下文示例和测试示例之间的交互
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config

        # 标签编码器（用于分类和回归任务）
        self.label_encoder = LabelEncoder(config)

        # 上下文-查询交互层
        self.num_icl_layers = 4
        self.icl_layers = nn.ModuleList([
            ICLLayer(config) for _ in range(self.num_icl_layers)
        ])

        # 任务头
        self.task_heads = nn.ModuleDict({
            'classification': ClassificationHead(config),
            'regression': RegressionHead(config),
            'link_prediction': LinkPredictionHead(config)
        })

        # 上下文聚合策略
        self.context_aggregation = config.context_aggregation  # attention, mean, weighted

        if self.context_aggregation == "attention":
            self.context_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.attention_dropout,
                batch_first=True
            )

    def forward(self,
                query_embeddings: torch.Tensor,
                context_embeddings: torch.Tensor,
                context_labels: torch.Tensor,
                task_type: str,
                label_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        query_embeddings: (batch_size, hidden_dim) 查询图嵌入
        context_embeddings: (batch_size, num_context, hidden_dim) 上下文图嵌入
        context_labels: (batch_size, num_context, ...) 上下文标签
        task_type: 任务类型
        label_metadata: 标签元数据（如类别数等）

        返回: {
            'predictions': 预测结果,
            'attention_weights': 注意力权重（如果使用注意力聚合）
        }
        """
        batch_size = query_embeddings.shape[0]
        num_context = context_embeddings.shape[1]

        # 编码标签
        encoded_labels = self.label_encoder(context_labels, task_type, label_metadata)

        # 将标签信息融合到上下文嵌入中
        context_with_labels = self._fuse_labels_with_context(
            context_embeddings, encoded_labels
        )

        # 扩展查询维度以匹配批处理
        query_expanded = query_embeddings.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # 通过ICL层处理
        for layer in self.icl_layers:
            query_expanded, context_with_labels = layer(
                query_expanded, context_with_labels
            )

        # 聚合上下文信息
        aggregated_context = self._aggregate_context(
            query_expanded.squeeze(1), context_with_labels
        )

        # 最终预测
        query_final = query_expanded.squeeze(1) + aggregated_context

        # 应用任务头
        outputs = self.task_heads[task_type](
            query_final,
            context_embeddings=context_with_labels,
            context_labels=context_labels,
            label_metadata=label_metadata
        )

        return outputs

    def _fuse_labels_with_context(self, context_embeddings: torch.Tensor,
                                  encoded_labels: torch.Tensor) -> torch.Tensor:
        """
        将编码的标签与上下文嵌入融合
        """
        # 简单相加（可以替换为更复杂的融合方式）
        return context_embeddings + encoded_labels

    def _aggregate_context(self, query: torch.Tensor,
                           context: torch.Tensor) -> torch.Tensor:
        """
        聚合上下文信息
        """
        if self.context_aggregation == "mean":
            return context.mean(dim=1)

        elif self.context_aggregation == "attention":
            # 使用查询对上下文进行注意力聚合
            query = query.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            aggregated, attention_weights = self.context_attention(
                query, context, context
            )
            return aggregated.squeeze(1)

        elif self.context_aggregation == "weighted":
            # 基于相似度的加权聚合
            similarities = F.cosine_similarity(
                query.unsqueeze(1), context, dim=-1
            )  # (batch_size, num_context)

            weights = F.softmax(similarities, dim=-1)
            aggregated = torch.bmm(
                weights.unsqueeze(1), context
            ).squeeze(1)

            return aggregated

        else:
            raise ValueError(f"未知的聚合策略: {self.context_aggregation}")


class ICLLayer(nn.Module):
    """
    单个ICL层
    实现查询和上下文之间的双向交互
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()

        # 查询自注意力
        self.query_self_attention = SelfAttentionBlock(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )

        # 上下文自注意力
        self.context_self_attention = SelfAttentionBlock(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )

        # 查询到上下文的交叉注意力
        self.query_to_context = CrossAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )

        # 上下文到查询的交叉注意力（可选）
        self.bidirectional = True
        if self.bidirectional:
            self.context_to_query = CrossAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.attention_dropout
            )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: (batch_size, 1, hidden_dim)
        context: (batch_size, num_context, hidden_dim)

        返回: (更新的查询, 更新的上下文)
        """
        # 自注意力
        query = self.query_self_attention(query)
        context = self.context_self_attention(context)

        # 查询关注上下文
        query_updated = self.query_to_context(query, context)

        # 双向注意力
        if self.bidirectional:
            context_updated = self.context_to_query(context, query)
        else:
            context_updated = context

        return query_updated, context_updated


class LabelEncoder(nn.Module):
    """
    标签编码器
    将不同类型的标签编码为统一的表示
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config

        # 分类标签编码
        self.class_embedding = nn.Embedding(
            1000,  # 最大类别数
            config.hidden_dim
        )

        # 回归标签编码
        self.regression_encoder = nn.Sequential(
            nn.Linear(1, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )

        # 多标签编码
        self.multilabel_encoder = nn.Linear(
            1000,  # 最大标签数
            config.hidden_dim
        )

    def forward(self, labels: torch.Tensor, task_type: str,
                metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        labels: 标签张量，形状取决于任务类型
        task_type: classification, regression, multilabel, link_prediction
        metadata: 元数据

        返回: (batch_size, num_context, hidden_dim)
        """
        batch_size, num_context = labels.shape[:2]

        if task_type == "classification":
            # 分类任务：labels shape (batch_size, num_context)
            encoded = self.class_embedding(labels.long())

        elif task_type == "regression":
            # 回归任务：labels shape (batch_size, num_context) 或 (batch_size, num_context, 1)
            if labels.dim() == 2:
                labels = labels.unsqueeze(-1)
            encoded = self.regression_encoder(labels)

        elif task_type == "multilabel":
            # 多标签任务：labels shape (batch_size, num_context, num_labels)
            # 转换为one-hot或多热编码
            if metadata and 'num_labels' in metadata:
                num_labels = metadata['num_labels']
                if labels.shape[-1] != num_labels:
                    # 假设labels是索引，需要转换为多热编码
                    labels_multihot = torch.zeros(
                        batch_size, num_context, num_labels,
                        device=labels.device
                    )
                    labels_multihot.scatter_(-1, labels.long(), 1)
                    labels = labels_multihot

            encoded = self.multilabel_encoder(labels.float())

        elif task_type == "link_prediction":
            # 链接预测：labels是目标节点ID列表
            # 这里简化处理，实际应该使用目标节点的嵌入
            if metadata and 'node_embeddings' in metadata:
                # 使用提供的节点嵌入
                node_embeddings = metadata['node_embeddings']
                encoded = torch.stack([
                    node_embeddings[labels[i].long()]
                    for i in range(batch_size)
                ])
            else:
                # 使用ID嵌入作为后备
                encoded = self.class_embedding(labels.long())

        else:
            raise ValueError(f"未知的任务类型: {task_type}")

        return encoded


class ClassificationHead(nn.Module):
    """分类任务头"""

    def __init__(self, config: KumoRFMConfig):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # 原型学习
        self.use_prototypes = True

    def forward(self, query: torch.Tensor,
                context_embeddings: torch.Tensor,
                context_labels: torch.Tensor,
                label_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        使用原型学习进行分类
        """
        # 投影查询
        query_proj = self.projection(query)

        if self.use_prototypes:
            # 计算每个类别的原型（平均嵌入）
            num_classes = label_metadata.get('num_classes',
                                             context_labels.max().item() + 1)

            prototypes = []
            for c in range(num_classes):
                # 找到属于类别c的所有上下文示例
                mask = (context_labels == c)
                if mask.any():
                    class_embeddings = context_embeddings[mask]
                    prototype = class_embeddings.mean(dim=0)
                else:
                    # 如果没有该类别的示例，使用零向量
                    prototype = torch.zeros_like(query_proj[0])

                prototypes.append(prototype)

            prototypes = torch.stack(prototypes)  # (num_classes, hidden_dim)

            # 计算与原型的相似度
            similarities = F.cosine_similarity(
                query_proj.unsqueeze(1),
                prototypes.unsqueeze(0),
                dim=-1
            )  # (batch_size, num_classes)

            # 转换为logits
            logits = similarities * 10  # 温度缩放

        else:
            # 直接预测
            logits = self.projection(query_proj)

        return {
            'logits': logits,
            'predictions': logits.argmax(dim=-1)
        }


class RegressionHead(nn.Module):
    """回归任务头"""

    def __init__(self, config: KumoRFMConfig):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        # 是否使用上下文统计信息
        self.use_context_stats = True

    def forward(self, query: torch.Tensor,
                context_embeddings: Optional[torch.Tensor] = None,
                context_labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        回归预测
        """
        # 基础预测
        predictions = self.projection(query).squeeze(-1)

        if self.use_context_stats and context_labels is not None:
            # 使用上下文标签的统计信息进行调整
            context_mean = context_labels.float().mean(dim=1)
            context_std = context_labels.float().std(dim=1)

            # 标准化预测
            predictions = predictions * context_std + context_mean

        return {
            'predictions': predictions,
            'logits': predictions  # 为了接口一致性
        }


class LinkPredictionHead(nn.Module):
    """链接预测任务头"""

    def __init__(self, config: KumoRFMConfig):
        super().__init__()

        # 用户和物品的投影
        self.user_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.item_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

        # 评分预测
        self.score_projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, query: torch.Tensor,
                context_embeddings: Optional[torch.Tensor] = None,
                context_labels: Optional[torch.Tensor] = None,
                label_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        链接预测（如推荐任务）
        """
        if label_metadata and 'candidate_embeddings' in label_metadata:
            # 有候选物品嵌入
            candidates = label_metadata['candidate_embeddings']
            num_candidates = candidates.shape[1]

            # 投影用户和物品
            user_emb = self.user_projection(query)  # (batch_size, hidden_dim)
            item_emb = self.item_projection(candidates)  # (batch_size, num_candidates, hidden_dim)

            # 计算分数
            # 方法1：点积
            scores = torch.bmm(item_emb, user_emb.unsqueeze(-1)).squeeze(-1)

            # 方法2：MLP
            # user_expanded = user_emb.unsqueeze(1).expand(-1, num_candidates, -1)
            # combined = torch.cat([user_expanded, item_emb], dim=-1)
            # scores = self.score_projection(combined).squeeze(-1)

            # 获取top-k
            k = label_metadata.get('k', 10)
            top_scores, top_indices = scores.topk(k, dim=-1)

            return {
                'scores': scores,
                'predictions': top_indices,
                'top_scores': top_scores
            }

        else:
            # 没有候选集，返回用户嵌入
            return {
                'predictions': self.user_projection(query),
                'logits': query
            }