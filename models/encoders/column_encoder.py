"""
列编码器
支持多种数据类型的列编码，包括数值、分类、文本、时间戳和嵌入
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Any
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from datetime import datetime

from config.model_config import ColumnEncoderConfig


class NumericalEncoder(nn.Module):
    """数值列编码器"""

    def __init__(self, config: ColumnEncoderConfig):
        super().__init__()
        self.config = config

        # 数值嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(1, config.numerical_embedding_dim),
            nn.ReLU(),
            nn.Linear(config.numerical_embedding_dim, config.numerical_embedding_dim)
        )

        # 可学习的归一化参数
        if config.use_numerical_normalization:
            self.mean = nn.Parameter(torch.zeros(1))
            self.std = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch_size, seq_len) 或 (batch_size,)
        mask: 指示有效值的掩码
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # 处理缺失值
        if mask is not None:
            x = x.masked_fill(~mask, 0.0)

        # 归一化
        if self.config.use_numerical_normalization:
            x = (x - self.mean) / (self.std + 1e-6)

        # 嵌入
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        embeddings = self.embedding(x)

        return embeddings


class CategoricalEncoder(nn.Module):
    """分类列编码器"""

    def __init__(self, config: ColumnEncoderConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # 嵌入层（+1 是为了未知标记）
        self.embedding = nn.Embedding(
            vocab_size + 1,
            config.categorical_embedding_dim,
            padding_idx=config.unknown_token_id
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len) 或 (batch_size,)
        """
        # 处理超出词汇表的值
        x = torch.where(
            x >= self.vocab_size,
            torch.tensor(self.config.unknown_token_id, device=x.device),
            x
        )

        embeddings = self.embedding(x)
        return embeddings


class TextEncoder(nn.Module):
    """文本列编码器"""

    def __init__(self, config: ColumnEncoderConfig, model_name: str):
        super().__init__()
        self.config = config

        # 加载预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)

        # 投影层（如果需要）
        if self.text_model.config.hidden_size != config.text_embedding_dim:
            self.projection = nn.Linear(
                self.text_model.config.hidden_size,
                config.text_embedding_dim
            )
        else:
            self.projection = None

        # 冻结文本模型参数（可选）
        for param in self.text_model.parameters():
            param.requires_grad = False

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        texts: 文本列表
        """
        # 分词
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt"
        )

        # 编码
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.text_model(**inputs)

        # 使用CLS标记的表示
        embeddings = outputs.last_hidden_state[:, 0, :]

        # 投影
        if self.projection is not None:
            embeddings = self.projection(embeddings)

        return embeddings


class TimeEncoder(nn.Module):
    """时间戳编码器"""

    def __init__(self, config: ColumnEncoderConfig):
        super().__init__()
        self.config = config

        if config.time_encoding_type == "sinusoidal":
            # 正弦编码
            self.use_sinusoidal = True
            # 时间特征数：年、月、日、星期、小时等
            self.time_features = 7
            self.projection = nn.Linear(
                self.time_features * 2,  # sin和cos
                config.time_embedding_dim
            )
        else:
            # 可学习编码
            self.use_sinusoidal = False
            self.embedding = nn.Linear(7, config.time_embedding_dim)

    def extract_time_features(self, timestamps: List[datetime]) -> torch.Tensor:
        """提取时间特征"""
        features = []

        for ts in timestamps:
            if pd.isna(ts):
                # 缺失值处理
                feat = [0] * self.time_features
            else:
                # 转换为pandas时间戳
                if not isinstance(ts, pd.Timestamp):
                    ts = pd.Timestamp(ts)

                # 提取特征
                feat = [
                    ts.year / 2025.0,  # 归一化年份
                    ts.month / 12.0,  # 月份
                    ts.day / 31.0,  # 日期
                    ts.dayofweek / 6.0,  # 星期
                    ts.hour / 23.0,  # 小时
                    ts.dayofyear / 365.0,  # 一年中的第几天
                    ts.quarter / 4.0  # 季度
                ]

            features.append(feat)

        return torch.tensor(features, dtype=torch.float32)

    def forward(self, timestamps: List[datetime]) -> torch.Tensor:
        """
        timestamps: 时间戳列表
        """
        # 提取时间特征
        time_features = self.extract_time_features(timestamps)

        if self.use_sinusoidal:
            # 正弦编码
            sin_features = torch.sin(2 * np.pi * time_features)
            cos_features = torch.cos(2 * np.pi * time_features)
            features = torch.cat([sin_features, cos_features], dim=-1)
            embeddings = self.projection(features)
        else:
            # 可学习编码
            embeddings = self.embedding(time_features)

        return embeddings


class EmbeddingEncoder(nn.Module):
    """嵌入列编码器（用于处理自定义上游嵌入）"""

    def __init__(self, config: ColumnEncoderConfig, input_dim: int):
        super().__init__()
        self.config = config

        if config.embedding_projection_dim and input_dim != config.embedding_projection_dim:
            # 需要投影
            self.projection = nn.Linear(input_dim, config.embedding_projection_dim)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_dim) 或 (batch_size, seq_len, input_dim)
        """
        if self.projection is not None:
            x = self.projection(x)

        return x


class MultiModalColumnEncoder(nn.Module):
    """
    多模态列编码器
    管理所有类型的列编码器
    """

    def __init__(self, config: ColumnEncoderConfig,
                 column_metadata: Dict[str, Dict[str, Any]],
                 text_model_name: Optional[str] = None):
        """
        column_metadata: 列元数据，格式为 {列名: {dtype: 类型, vocab_size: 大小, ...}}
        """
        super().__init__()
        self.config = config
        self.column_metadata = column_metadata

        # 为每个列创建相应的编码器
        self.encoders = nn.ModuleDict()
        self.column_to_encoder = {}

        for col_name, metadata in column_metadata.items():
            dtype = metadata['dtype']

            if dtype == 'numerical':
                encoder = NumericalEncoder(config)

            elif dtype == 'categorical':
                vocab_size = metadata.get('vocab_size', 100)
                encoder = CategoricalEncoder(config, vocab_size)

            elif dtype == 'text':
                model_name = text_model_name or config.text_encoder_model
                encoder = TextEncoder(config, model_name)

            elif dtype == 'timestamp':
                encoder = TimeEncoder(config)

            elif dtype == 'embedding':
                input_dim = metadata.get('embedding_dim', 128)
                encoder = EmbeddingEncoder(config, input_dim)

            else:
                raise ValueError(f"未知的列类型: {dtype}")

            # 使用列名和类型的组合作为键，避免重名
            encoder_key = f"{col_name}_{dtype}"
            self.encoders[encoder_key] = encoder
            self.column_to_encoder[col_name] = encoder_key

    def get_output_dim(self) -> int:
        """获取输出维度"""
        # 所有编码器的输出维度应该相同
        dims = set()

        for col_name, metadata in self.column_metadata.items():
            dtype = metadata['dtype']

            if dtype == 'numerical':
                dims.add(self.config.numerical_embedding_dim)
            elif dtype == 'categorical':
                dims.add(self.config.categorical_embedding_dim)
            elif dtype == 'text':
                dims.add(self.config.text_embedding_dim)
            elif dtype == 'timestamp':
                dims.add(self.config.time_embedding_dim)
            elif dtype == 'embedding':
                dims.add(self.config.embedding_projection_dim or metadata.get('embedding_dim', 128))

        if len(dims) > 1:
            raise ValueError(f"编码器输出维度不一致: {dims}")

        return dims.pop() if dims else 0

    def forward(self, column_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        column_data: {列名: 数据}
        返回: {列名: 编码后的张量}
        """
        encoded_columns = {}

        for col_name, data in column_data.items():
            if col_name not in self.column_to_encoder:
                continue

            encoder_key = self.column_to_encoder[col_name]
            encoder = self.encoders[encoder_key]

            # 编码
            encoded = encoder(data)
            encoded_columns[col_name] = encoded

        return encoded_columns