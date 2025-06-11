"""
表级编码器
将多个列的编码聚合成行级表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from config.model_config import KumoRFMConfig


class TableTransformer(nn.Module):
    """
    表级Transformer
    在列维度上应用注意力机制，生成行级表示
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config

        # 列位置编码
        self.column_position_embedding = nn.Embedding(
            config.max_columns,
            config.hidden_dim
        )

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.attention_dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3  # 表级使用较少的层数
        )

        # 输出投影
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

        # 列类型嵌入（可选）
        self.column_type_embedding = nn.Embedding(
            5,  # 5种列类型
            config.hidden_dim
        )

        # CLS标记
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))

    def forward(self, column_embeddings: torch.Tensor,
                column_mask: Optional[torch.Tensor] = None,
                column_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        column_embeddings: (batch_size, num_columns, hidden_dim)
        column_mask: (batch_size, num_columns) 布尔掩码，True表示有效列
        column_types: (batch_size, num_columns) 列类型ID

        返回: (batch_size, hidden_dim) 行级表示
        """
        batch_size, num_columns, hidden_dim = column_embeddings.shape

        # 添加CLS标记
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        embeddings = torch.cat([cls_tokens, column_embeddings], dim=1)

        # 位置编码
        positions = torch.arange(num_columns + 1, device=embeddings.device)
        position_embeddings = self.column_position_embedding(positions)
        embeddings = embeddings + position_embeddings.unsqueeze(0)

        # 列类型编码（如果提供）
        if column_types is not None:
            # 为CLS标记添加特殊类型
            cls_type = torch.zeros(batch_size, 1, dtype=column_types.dtype, device=column_types.device)
            extended_types = torch.cat([cls_type, column_types], dim=1)
            type_embeddings = self.column_type_embedding(extended_types)
            embeddings = embeddings + type_embeddings

        # 创建注意力掩码
        if column_mask is not None:
            # 为CLS标记添加True
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=column_mask.device)
            extended_mask = torch.cat([cls_mask, column_mask], dim=1)

            # 转换为Transformer需要的格式（True位置会被忽略）
            attention_mask = ~extended_mask
        else:
            attention_mask = None

        # Transformer编码
        encoded = self.transformer(embeddings, src_key_padding_mask=attention_mask)

        # 使用CLS标记作为行表示
        row_representation = encoded[:, 0, :]

        # 输出投影
        output = self.output_projection(row_representation)

        return output


class TableEncoder(nn.Module):
    """
    表编码器
    处理表中的所有行，生成表级表示
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config

        # 表级Transformer
        self.table_transformer = TableTransformer(config)

        # 表类型嵌入
        self.table_type_embedding = nn.Embedding(
            100,  # 最多100种表类型
            config.hidden_dim
        )

        # 行聚合
        self.row_aggregation = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        # 输出归一化
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def aggregate_columns(self, column_embeddings_dict: Dict[str, torch.Tensor],
                          column_order: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将列字典转换为张量格式

        column_embeddings_dict: {列名: (batch_size, hidden_dim)}
        column_order: 列的顺序

        返回:
        - column_embeddings: (batch_size, num_columns, hidden_dim)
        - column_mask: (batch_size, num_columns)
        """
        batch_size = next(iter(column_embeddings_dict.values())).shape[0]
        hidden_dim = next(iter(column_embeddings_dict.values())).shape[-1]
        max_columns = min(len(column_order), self.config.max_columns)

        # 初始化张量
        column_embeddings = torch.zeros(
            batch_size, max_columns, hidden_dim,
            device=next(iter(column_embeddings_dict.values())).device
        )
        column_mask = torch.zeros(batch_size, max_columns, dtype=torch.bool)

        # 填充列嵌入
        for i, col_name in enumerate(column_order[:max_columns]):
            if col_name in column_embeddings_dict:
                column_embeddings[:, i, :] = column_embeddings_dict[col_name]
                column_mask[:, i] = True

        return column_embeddings, column_mask

    def forward(self, tables_data: Dict[str, Dict[str, torch.Tensor]],
                table_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        tables_data: {表名: {列名: 列嵌入}}
        table_metadata: {表名: {元数据}}

        返回: {表名: (num_rows, hidden_dim)}
        """
        table_representations = {}

        for table_name, column_embeddings_dict in tables_data.items():
            if table_name not in table_metadata:
                continue

            metadata = table_metadata[table_name]
            column_order = metadata.get('column_order', list(column_embeddings_dict.keys()))

            # 聚合列嵌入
            column_embeddings, column_mask = self.aggregate_columns(
                column_embeddings_dict, column_order
            )

            # 生成行表示
            if column_embeddings.shape[0] > 0:
                row_representations = self.table_transformer(
                    column_embeddings, column_mask
                )

                # 添加表类型嵌入
                if 'table_type_id' in metadata:
                    table_type_id = metadata['table_type_id']
                    table_type_emb = self.table_type_embedding(
                        torch.tensor(table_type_id, device=row_representations.device)
                    )
                    row_representations = row_representations + table_type_emb

                # 归一化
                row_representations = self.layer_norm(row_representations)

                table_representations[table_name] = row_representations

        return table_representations


class RowAggregator(nn.Module):
    """
    行聚合器
    将同一表的多行聚合成一个表示（用于处理一对多关系）
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config

        # 聚合方式
        self.aggregation_type = "attention"  # mean, max, attention

        if self.aggregation_type == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.attention_dropout,
                batch_first=True
            )

            # 查询向量
            self.query = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))

    def forward(self, row_embeddings: torch.Tensor,
                row_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        row_embeddings: (batch_size, num_rows, hidden_dim)
        row_mask: (batch_size, num_rows) 布尔掩码

        返回: (batch_size, hidden_dim)
        """
        if self.aggregation_type == "mean":
            if row_mask is not None:
                # 掩码平均
                row_embeddings = row_embeddings * row_mask.unsqueeze(-1)
                sum_embeddings = row_embeddings.sum(dim=1)
                count = row_mask.sum(dim=1, keepdim=True).clamp(min=1)
                return sum_embeddings / count
            else:
                return row_embeddings.mean(dim=1)

        elif self.aggregation_type == "max":
            if row_mask is not None:
                # 掩码最大值
                row_embeddings = row_embeddings.masked_fill(
                    ~row_mask.unsqueeze(-1), float('-inf')
                )
            return row_embeddings.max(dim=1)[0]

        elif self.aggregation_type == "attention":
            batch_size = row_embeddings.shape[0]

            # 扩展查询向量
            queries = self.query.expand(batch_size, 1, -1)

            # 注意力聚合
            if row_mask is not None:
                key_padding_mask = ~row_mask
            else:
                key_padding_mask = None

            attended, _ = self.attention(
                queries, row_embeddings, row_embeddings,
                key_padding_mask=key_padding_mask
            )

            return attended.squeeze(1)

        else:
            raise ValueError(f"未知的聚合类型: {self.aggregation_type}")