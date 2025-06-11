"""
注意力机制模块
包括多头注意力和图注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"隐藏维度 {hidden_dim} 必须能被注意力头数 {num_heads} 整除"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 查询、键、值投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        query: (batch_size, seq_len_q, hidden_dim)
        key: (batch_size, seq_len_k, hidden_dim)
        value: (batch_size, seq_len_v, hidden_dim)
        mask: (batch_size, seq_len_q, seq_len_k) 或 (seq_len_q, seq_len_k)

        返回:
        - output: (batch_size, seq_len_q, hidden_dim)
        - attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k) 如果return_attention=True
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]

        # 投影并重塑为多头
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len_v, self.num_heads, self.head_dim)

        # 转置以便于计算注意力
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len_q, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len_k, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_len_v, head_dim)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 应用掩码
        if mask is not None:
            if mask.dim() == 2:
                # 扩展掩码维度
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # 应用注意力权重
        attention_output = torch.matmul(attention_weights, V)

        # 重塑并投影输出
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len_q, self.hidden_dim)
        output = self.out_proj(attention_output)
        output = self.output_dropout(output)

        if return_attention:
            return output, attention_weights
        else:
            return output


class GraphAttention(nn.Module):
    """
    图注意力机制
    考虑图结构的注意力
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1,
                 use_edge_features: bool = False, edge_dim: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_edge_features = use_edge_features

        # 节点特征投影
        self.node_proj = nn.Linear(hidden_dim, hidden_dim * 2)  # 用于计算注意力系数

        # 边特征投影（如果使用）
        if use_edge_features and edge_dim:
            self.edge_proj = nn.Linear(edge_dim, num_heads)

        # 注意力参数
        self.attention_weights = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attention_weights)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        node_features: (num_nodes, hidden_dim)
        edge_index: (2, num_edges)
        edge_features: (num_edges, edge_dim) 可选

        返回: (num_nodes, hidden_dim)
        """
        num_nodes = node_features.shape[0]

        # 投影节点特征
        node_features_proj = self.node_proj(node_features)
        node_features_proj = node_features_proj.view(num_nodes, self.num_heads, 2 * self.head_dim)

        # 分离源节点和目标节点特征
        row, col = edge_index
        src_features = node_features_proj[row]  # (num_edges, num_heads, 2 * head_dim)
        dst_features = node_features_proj[col]

        # 计算注意力系数
        edge_features_cat = torch.cat([src_features, dst_features], dim=-1)
        attention_scores = (edge_features_cat * self.attention_weights).sum(dim=-1)

        # 添加边特征（如果有）
        if self.use_edge_features and edge_features is not None:
            edge_scores = self.edge_proj(edge_features)  # (num_edges, num_heads)
            attention_scores = attention_scores + edge_scores

        # LeakyReLU
        attention_scores = self.leaky_relu(attention_scores)

        # Softmax（对每个节点的所有入边）
        attention_weights = self._sparse_softmax(attention_scores, col, num_nodes)
        attention_weights = self.dropout(attention_weights)

        # 聚合邻居特征
        node_features_multi = node_features.view(num_nodes, self.num_heads, self.head_dim)
        src_features_weighted = node_features_multi[row] * attention_weights.unsqueeze(-1)

        # 累加到目标节点
        output = torch.zeros_like(node_features_multi)
        output.index_add_(0, col, src_features_weighted)

        # 重塑并投影
        output = output.view(num_nodes, self.hidden_dim)
        output = self.out_proj(output)

        return output

    def _sparse_softmax(self, attention_scores: torch.Tensor,
                        indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        稀疏Softmax
        对每个节点的入边进行softmax
        """
        # 获取每个节点的最大分数（数值稳定性）
        max_scores = torch.zeros(num_nodes, self.num_heads, device=attention_scores.device)
        max_scores.index_reduce_(0, indices, attention_scores, 'amax')

        # 减去最大值并计算exp
        attention_scores = attention_scores - max_scores[indices]
        exp_scores = attention_scores.exp()

        # 计算每个节点的归一化因子
        sum_exp = torch.zeros(num_nodes, self.num_heads, device=exp_scores.device)
        sum_exp.index_add_(0, indices, exp_scores)

        # 归一化
        normalized_scores = exp_scores / (sum_exp[indices] + 1e-10)

        return normalized_scores


class CrossAttention(nn.Module):
    """
    交叉注意力机制
    用于上下文学习中的查询-上下文交互
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 层归一化
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query: (batch_size, query_len, hidden_dim) 查询
        context: (batch_size, context_len, hidden_dim) 上下文
        context_mask: (batch_size, context_len) 上下文掩码

        返回: (batch_size, query_len, hidden_dim)
        """
        # 层归一化
        query_norm = self.norm_q(query)
        context_norm = self.norm_kv(context)

        # 交叉注意力
        if context_mask is not None:
            # 扩展掩码维度 (batch_size, 1, context_len)
            attention_mask = context_mask.unsqueeze(1).expand(-1, query.shape[1], -1)
        else:
            attention_mask = None

        attended = self.multihead_attention(
            query=query_norm,
            key=context_norm,
            value=context_norm,
            mask=attention_mask
        )

        # 残差连接
        query = query + attended

        # 前馈网络
        query_norm = self.norm_ffn(query)
        query = query + self.ffn(query_norm)

        return query


class SelfAttentionBlock(nn.Module):
    """
    自注意力块
    包含自注意力、层归一化和前馈网络
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch_size, seq_len, hidden_dim)
        mask: (batch_size, seq_len) 或 (batch_size, seq_len, seq_len)

        返回: (batch_size, seq_len, hidden_dim)
        """
        # 自注意力
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, x, x, mask=mask)
        x = residual + x

        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x