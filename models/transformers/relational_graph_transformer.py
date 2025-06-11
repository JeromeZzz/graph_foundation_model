"""
关系图Transformer
在异构图上执行表级注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math

from config.model_config import KumoRFMConfig
from .attention import MultiHeadAttention, GraphAttention


class RelationalGraphTransformerLayer(nn.Module):
    """
    单层关系图Transformer
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config

        # 自注意力
        self.self_attention = MultiHeadAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)

        # 表特定的参数（可选）
        self.use_table_specific_params = True
        if self.use_table_specific_params:
            # 为每种表类型创建特定的查询/键/值投影
            self.table_specific_qkv = nn.ModuleDict()

    def forward(self, node_features: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                node_types: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        node_features: (batch_size, num_nodes, hidden_dim)
        edge_index: (2, num_edges) 边索引
        node_types: (batch_size, num_nodes) 节点类型
        attention_mask: (batch_size, num_nodes, num_nodes) 注意力掩码

        返回: (batch_size, num_nodes, hidden_dim)
        """
        # 自注意力
        residual = node_features
        node_features = self.norm1(node_features)

        # 如果提供了边索引，创建基于图结构的注意力掩码
        if edge_index is not None and attention_mask is None:
            attention_mask = self._create_graph_attention_mask(
                edge_index, node_features.shape[1], node_features.device
            )

        # 应用注意力
        attended_features = self.self_attention(
            query=node_features,
            key=node_features,
            value=node_features,
            mask=attention_mask
        )

        node_features = residual + self.dropout(attended_features)

        # 前馈网络
        residual = node_features
        node_features = self.norm2(node_features)
        node_features = self.feed_forward(node_features)
        node_features = residual + self.dropout(node_features)

        return node_features

    def _create_graph_attention_mask(self, edge_index: torch.Tensor,
                                     num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        基于图结构创建注意力掩码
        只允许连接的节点之间进行注意力
        """
        # 创建邻接矩阵
        row, col = edge_index
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=device)
        adj_matrix[row, col] = 1
        adj_matrix[col, row] = 1  # 无向图

        # 添加自环
        adj_matrix.fill_diagonal_(1)

        # 转换为注意力掩码（0表示允许注意力，-inf表示阻止）
        attention_mask = torch.where(
            adj_matrix > 0,
            torch.tensor(0.0, device=device),
            torch.tensor(float('-inf'), device=device)
        )

        return attention_mask.unsqueeze(0)  # 添加batch维度


class RelationalGraphTransformer(nn.Module):
    """
    完整的关系图Transformer
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config

        # 多层Transformer
        self.layers = nn.ModuleList([
            RelationalGraphTransformerLayer(config)
            for _ in range(config.num_layers)
        ])

        # 最终层归一化
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # 池化策略
        self.pooling_strategy = "center"  # center, mean, max, attention

        if self.pooling_strategy == "attention":
            # 注意力池化
            self.pooling_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.attention_dropout,
                batch_first=True
            )
            self.pooling_query = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))

    def forward(self, node_features: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                node_types: Optional[torch.Tensor] = None,
                center_node_idx: Optional[torch.Tensor] = None,
                batch_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        node_features: (total_nodes, hidden_dim) 或 (batch_size, num_nodes, hidden_dim)
        edge_index: (2, num_edges) 边索引
        node_types: (total_nodes,) 或 (batch_size, num_nodes) 节点类型
        center_node_idx: (batch_size,) 每个图的中心节点索引
        batch_idx: (total_nodes,) 节点到批次的映射

        返回: {
            'node_embeddings': 所有节点的嵌入,
            'graph_embedding': 图级嵌入,
            'center_embedding': 中心节点嵌入
        }
        """
        # 判断输入格式
        if node_features.dim() == 2:
            # 扁平化格式，需要batch_idx
            is_batched = False
            total_nodes = node_features.shape[0]
        else:
            # 批次格式
            is_batched = True
            batch_size, num_nodes, hidden_dim = node_features.shape
            node_features = node_features.view(-1, hidden_dim)
            total_nodes = batch_size * num_nodes

        # 通过所有层
        for layer in self.layers:
            if is_batched:
                # 重塑为批次格式
                node_features = node_features.view(batch_size, num_nodes, -1)
                node_features = layer(node_features, edge_index, node_types)
                node_features = node_features.view(-1, self.config.hidden_dim)
            else:
                # 扁平格式处理
                node_features = layer(
                    node_features.unsqueeze(0),
                    edge_index,
                    node_types
                ).squeeze(0)

        # 最终归一化
        node_features = self.final_norm(node_features)

        # 准备输出
        outputs = {'node_embeddings': node_features}

        # 提取图级表示
        if is_batched:
            node_features = node_features.view(batch_size, num_nodes, -1)

            if self.pooling_strategy == "center" and center_node_idx is not None:
                # 使用中心节点作为图表示
                batch_indices = torch.arange(batch_size, device=node_features.device)
                center_embeddings = node_features[batch_indices, center_node_idx]
                outputs['graph_embedding'] = center_embeddings
                outputs['center_embedding'] = center_embeddings

            elif self.pooling_strategy == "mean":
                # 平均池化
                outputs['graph_embedding'] = node_features.mean(dim=1)

            elif self.pooling_strategy == "max":
                # 最大池化
                outputs['graph_embedding'] = node_features.max(dim=1)[0]

            elif self.pooling_strategy == "attention":
                # 注意力池化
                query = self.pooling_query.expand(batch_size, 1, -1)
                attended, _ = self.pooling_attention(
                    query, node_features, node_features
                )
                outputs['graph_embedding'] = attended.squeeze(1)

        else:
            # 扁平格式，使用batch_idx进行池化
            if batch_idx is not None:
                batch_size = batch_idx.max().item() + 1
                graph_embeddings = []

                for b in range(batch_size):
                    mask = batch_idx == b
                    batch_nodes = node_features[mask]

                    if self.pooling_strategy == "center" and center_node_idx is not None:
                        graph_emb = batch_nodes[center_node_idx[b]]
                    elif self.pooling_strategy == "mean":
                        graph_emb = batch_nodes.mean(dim=0)
                    elif self.pooling_strategy == "max":
                        graph_emb = batch_nodes.max(dim=0)[0]
                    else:
                        graph_emb = batch_nodes.mean(dim=0)  # 默认平均

                    graph_embeddings.append(graph_emb)

                outputs['graph_embedding'] = torch.stack(graph_embeddings)

        return outputs


class HeterogeneousRelationalGraphTransformer(nn.Module):
    """
    异构关系图Transformer
    支持多种节点类型和边类型
    """

    def __init__(self, config: KumoRFMConfig, num_node_types: int, num_edge_types: int):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # 基础Transformer
        self.transformer = RelationalGraphTransformer(config)

        # 节点类型特定的投影（可选）
        self.node_type_projections = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(num_node_types)
        ])

        # 边类型特定的注意力权重（可选）
        self.edge_type_weights = nn.Parameter(
            torch.ones(num_edge_types, config.num_heads)
        )

    def forward(self, node_features_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                num_nodes_dict: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        异构图的前向传播

        node_features_dict: {节点类型: 特征张量}
        edge_index_dict: {(源类型, 边类型, 目标类型): 边索引}
        num_nodes_dict: {节点类型: 节点数}

        返回: {节点类型: 更新后的特征}
        """
        # 合并所有节点特征
        all_node_features = []
        node_type_ids = []
        node_type_offsets = {}
        current_offset = 0

        for node_type, features in node_features_dict.items():
            all_node_features.append(features)
            node_type_offsets[node_type] = current_offset

            # 创建节点类型ID
            type_id = list(node_features_dict.keys()).index(node_type)
            node_type_ids.extend([type_id] * features.shape[0])

            current_offset += features.shape[0]

        # 拼接所有节点
        all_node_features = torch.cat(all_node_features, dim=0)
        node_type_ids = torch.tensor(node_type_ids, device=all_node_features.device)

        # 应用节点类型特定的投影
        projected_features = []
        for i in range(self.num_node_types):
            mask = node_type_ids == i
            if mask.any():
                type_features = all_node_features[mask]
                projected = self.node_type_projections[i](type_features)
                projected_features.append((mask, projected))

        # 重组投影后的特征
        for mask, features in projected_features:
            all_node_features[mask] = features

        # 合并所有边
        all_edges = []
        for (src_type, edge_type, dst_type), edge_index in edge_index_dict.items():
            # 添加偏移量
            src_offset = node_type_offsets[src_type]
            dst_offset = node_type_offsets[dst_type]

            offset_edge_index = edge_index.clone()
            offset_edge_index[0] += src_offset
            offset_edge_index[1] += dst_offset

            all_edges.append(offset_edge_index)

        if all_edges:
            all_edge_index = torch.cat(all_edges, dim=1)
        else:
            all_edge_index = None

        # 应用Transformer
        outputs = self.transformer(
            all_node_features,
            all_edge_index,
            node_type_ids
        )

        # 分离输出
        updated_features_dict = {}
        for node_type, num_nodes in num_nodes_dict.items():
            offset = node_type_offsets[node_type]
            updated_features_dict[node_type] = outputs['node_embeddings'][
                                               offset:offset + num_nodes
                                               ]

        return updated_features_dict