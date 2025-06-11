"""
位置编码模块
包括节点类型编码、跳数编码、时间编码和子图结构编码
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
from datetime import datetime

from config.model_config import KumoRFMConfig


class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:x.size(1)]


class NodeTypeEncoder(nn.Module):
    """
    节点类型编码器
    为不同表的节点添加类型标识
    """

    def __init__(self, config: KumoRFMConfig, num_node_types: int):
        super().__init__()
        self.config = config

        # 节点类型嵌入
        self.node_type_embedding = nn.Embedding(
            num_node_types,
            config.hidden_dim
        )

    def forward(self, node_types: torch.Tensor) -> torch.Tensor:
        """
        node_types: (batch_size, num_nodes) 节点类型ID
        返回: (batch_size, num_nodes, hidden_dim)
        """
        return self.node_type_embedding(node_types)


class HopEncoder(nn.Module):
    """
    跳数编码器
    编码节点到中心节点的距离
    """

    def __init__(self, config: KumoRFMConfig, max_hops: int = 3):
        super().__init__()
        self.config = config
        self.max_hops = max_hops

        # 跳数嵌入
        self.hop_embedding = nn.Embedding(
            max_hops + 1,  # 0跳表示中心节点
            config.hidden_dim
        )

    def forward(self, hop_distances: torch.Tensor) -> torch.Tensor:
        """
        hop_distances: (batch_size, num_nodes) 跳数距离
        返回: (batch_size, num_nodes, hidden_dim)
        """
        # 裁剪到最大跳数
        hop_distances = torch.clamp(hop_distances, max=self.max_hops)
        return self.hop_embedding(hop_distances)


class TemporalEncoder(nn.Module):
    """
    时间编码器
    编码相对时间差异
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config

        # 时间差异的不同粒度编码
        self.time_scales = [1, 7, 30, 365]  # 天、周、月、年

        # 时间编码投影
        self.time_projection = nn.Linear(
            len(self.time_scales) * 2,  # sin和cos
            config.hidden_dim
        )

    def encode_time_diff(self, time_diff_days: torch.Tensor) -> torch.Tensor:
        """
        编码时间差异
        time_diff_days: (batch_size, num_nodes) 时间差异（天数）
        """
        encodings = []

        for scale in self.time_scales:
            # 归一化时间差异
            normalized_diff = time_diff_days / scale

            # 正弦和余弦编码
            sin_encoding = torch.sin(2 * np.pi * normalized_diff)
            cos_encoding = torch.cos(2 * np.pi * normalized_diff)

            encodings.extend([sin_encoding, cos_encoding])

        # 拼接所有编码
        time_features = torch.stack(encodings, dim=-1)

        # 投影到隐藏维度
        return self.time_projection(time_features)

    def forward(self, node_timestamps: torch.Tensor,
                reference_timestamp: torch.Tensor) -> torch.Tensor:
        """
        node_timestamps: (batch_size, num_nodes) 节点时间戳（Unix时间）
        reference_timestamp: (batch_size,) 参考时间戳

        返回: (batch_size, num_nodes, hidden_dim)
        """
        # 计算时间差异（秒）
        time_diff_seconds = reference_timestamp.unsqueeze(1) - node_timestamps

        # 转换为天数
        time_diff_days = time_diff_seconds / (24 * 3600)

        # 编码
        return self.encode_time_diff(time_diff_days)


class SubgraphStructureEncoder(nn.Module):
    """
    子图结构编码器
    编码局部图结构信息（如度数、三角形数等）
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config

        # 结构特征数
        self.num_structure_features = 5  # 度数、入度、出度、三角形数、聚类系数

        # 结构特征编码器
        self.structure_encoder = nn.Sequential(
            nn.Linear(self.num_structure_features, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )

    def compute_structure_features(self, edge_index: torch.Tensor,
                                   num_nodes: int) -> torch.Tensor:
        """
        计算结构特征
        edge_index: (2, num_edges) 边索引
        num_nodes: 节点数

        返回: (num_nodes, num_structure_features)
        """
        device = edge_index.device

        # 初始化特征张量
        features = torch.zeros(num_nodes, self.num_structure_features, device=device)

        # 计算度数
        row, col = edge_index

        # 总度数
        degree = torch.zeros(num_nodes, device=device)
        degree.index_add_(0, row, torch.ones_like(row, dtype=torch.float))
        degree.index_add_(0, col, torch.ones_like(col, dtype=torch.float))
        features[:, 0] = degree

        # 入度
        in_degree = torch.zeros(num_nodes, device=device)
        in_degree.index_add_(0, col, torch.ones_like(col, dtype=torch.float))
        features[:, 1] = in_degree

        # 出度
        out_degree = torch.zeros(num_nodes, device=device)
        out_degree.index_add_(0, row, torch.ones_like(row, dtype=torch.float))
        features[:, 2] = out_degree

        # 三角形数（简化计算）
        # 这里使用一个近似：节点参与的边数越多，可能的三角形越多
        features[:, 3] = (degree * (degree - 1)) / 2

        # 聚类系数（简化版本）
        # 使用度数的倒数作为近似
        features[:, 4] = 1.0 / (degree + 1)

        return features

    def forward(self, edge_indices: List[torch.Tensor],
                num_nodes_list: List[int]) -> torch.Tensor:
        """
        edge_indices: 批次中每个图的边索引列表
        num_nodes_list: 每个图的节点数列表

        返回: (total_nodes, hidden_dim)
        """
        all_features = []

        for edge_index, num_nodes in zip(edge_indices, num_nodes_list):
            # 计算结构特征
            structure_features = self.compute_structure_features(edge_index, num_nodes)

            # 编码
            encoded_features = self.structure_encoder(structure_features)
            all_features.append(encoded_features)

        # 拼接所有图的特征
        return torch.cat(all_features, dim=0)


class CombinedPositionalEncoder(nn.Module):
    """
    组合位置编码器
    整合所有类型的位置编码
    """

    def __init__(self, config: KumoRFMConfig, num_node_types: int):
        super().__init__()
        self.config = config

        # 各种编码器
        self.encoders = nn.ModuleDict()

        if config.use_table_encoding:
            self.encoders['node_type'] = NodeTypeEncoder(config, num_node_types)

        if config.use_hop_encoding:
            self.encoders['hop'] = HopEncoder(config)

        if config.use_time_encoding:
            self.encoders['temporal'] = TemporalEncoder(config)

        if config.use_subgraph_encoding:
            self.encoders['structure'] = SubgraphStructureEncoder(config)

        # 组合方式
        self.combination = "sum"  # sum or concat

        if self.combination == "concat":
            # 投影层
            self.projection = nn.Linear(
                len(self.encoders) * config.hidden_dim,
                config.hidden_dim
            )

    def forward(self, positional_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        positional_info: 包含各种位置信息的字典
        - node_types: (batch_size, num_nodes)
        - hop_distances: (batch_size, num_nodes)
        - node_timestamps: (batch_size, num_nodes)
        - reference_timestamp: (batch_size,)
        - edge_indices: List[torch.Tensor]
        - num_nodes_list: List[int]

        返回: (batch_size, num_nodes, hidden_dim) 或 (total_nodes, hidden_dim)
        """
        encodings = []

        # 节点类型编码
        if 'node_type' in self.encoders and 'node_types' in positional_info:
            node_type_enc = self.encoders['node_type'](positional_info['node_types'])
            encodings.append(node_type_enc)

        # 跳数编码
        if 'hop' in self.encoders and 'hop_distances' in positional_info:
            hop_enc = self.encoders['hop'](positional_info['hop_distances'])
            encodings.append(hop_enc)

        # 时间编码
        if 'temporal' in self.encoders and 'node_timestamps' in positional_info:
            temporal_enc = self.encoders['temporal'](
                positional_info['node_timestamps'],
                positional_info['reference_timestamp']
            )
            encodings.append(temporal_enc)

        # 结构编码
        if 'structure' in self.encoders and 'edge_indices' in positional_info:
            structure_enc = self.encoders['structure'](
                positional_info['edge_indices'],
                positional_info['num_nodes_list']
            )
            encodings.append(structure_enc)

        # 组合编码
        if not encodings:
            return None

        if self.combination == "sum":
            # 直接相加
            combined = sum(encodings)
        elif self.combination == "concat":
            # 拼接后投影
            combined = torch.cat(encodings, dim=-1)
            combined = self.projection(combined)
        else:
            raise ValueError(f"未知的组合方式: {self.combination}")

        return combined