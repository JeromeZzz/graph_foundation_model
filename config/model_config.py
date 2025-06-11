"""
KumoRFM模型配置文件
定义所有模型相关的超参数和配置选项
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class KumoRFMConfig:
    """KumoRFM模型的主要配置类"""

    # 模型维度配置
    hidden_dim: int = 768  # 隐藏层维度
    num_heads: int = 12  # 注意力头数
    num_layers: int = 12  # Transformer层数

    # 编码器配置
    max_columns: int = 256  # 最大列数
    max_seq_length: int = 512  # 最大序列长度
    max_categorical_size: int = 50  # 分类变量的最大类别数
    text_encoder_model: str = "bert-base-uncased"  # 文本编码器

    # 图结构配置
    num_hops: int = 2  # 图采样跳数
    max_neighbors: int = 10  # 每跳最大邻居数
    temporal_sampling: str = "uniform"  # 时序采样策略: uniform, recent, fixed_interval

    # 上下文学习配置
    num_context_examples: int = 32  # 上下文示例数
    context_aggregation: str = "attention"  # 上下文聚合方式

    # 位置编码配置
    use_hop_encoding: bool = True  # 是否使用跳数编码
    use_time_encoding: bool = True  # 是否使用时间编码
    use_subgraph_encoding: bool = True  # 是否使用子图结构编码
    use_table_encoding: bool = True  # 是否使用表类型编码

    # 训练配置
    learning_rate: float = 1e-4
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Dropout配置
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1

    # 任务配置
    task_types: List[str] = None  # 支持的任务类型

    def __post_init__(self):
        """初始化后的验证和设置默认值"""
        if self.task_types is None:
            self.task_types = [
                "node_classification",
                "node_regression",
                "link_prediction"
            ]

        # 验证配置的合理性
        assert self.hidden_dim % self.num_heads == 0, \
            f"隐藏维度 {self.hidden_dim} 必须能被注意力头数 {self.num_heads} 整除"

        assert self.temporal_sampling in ["uniform", "recent", "fixed_interval"], \
            f"不支持的时序采样策略: {self.temporal_sampling}"


@dataclass
class ColumnEncoderConfig:
    """列编码器的配置"""

    # 数值列配置
    numerical_embedding_dim: int = 32
    use_numerical_normalization: bool = True

    # 分类列配置
    categorical_embedding_dim: int = 32
    unknown_token_id: int = 0

    # 时间戳列配置
    time_embedding_dim: int = 32
    time_encoding_type: str = "sinusoidal"  # sinusoidal or learned

    # 文本列配置
    text_embedding_dim: int = 768
    max_text_length: int = 128

    # 嵌入列配置（用于处理自定义上游嵌入）
    embedding_projection_dim: Optional[int] = None


@dataclass
class SamplerConfig:
    """图采样器的配置"""

    # 采样策略
    neighbor_sampling_strategy: str = "uniform"  # uniform, importance, temporal
    max_sampled_nodes: int = 500  # 单个子图的最大节点数

    # 时序采样
    lookback_window: Optional[int] = None  # 回看窗口（天数）
    lookahead_window: Optional[int] = None  # 前看窗口（天数）

    # 自适应采样
    use_adaptive_sampling: bool = True  # 冷启动场景下的自适应采样
    min_neighbors_per_hop: int = 1  # 每跳的最小邻居数