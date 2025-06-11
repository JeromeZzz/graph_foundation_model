"""
编码器模块
包含各种数据类型的编码器
"""

from .column_encoder import (
    NumericalEncoder,
    CategoricalEncoder,
    TextEncoder,
    TimeEncoder,
    EmbeddingEncoder,
    MultiModalColumnEncoder
)

from .table_encoder import (
    TableEncoder,
    TableTransformer,
    RowAggregator
)

from .positional_encoding import (
    PositionalEncoding,
    NodeTypeEncoder,
    HopEncoder,
    TemporalEncoder,
    SubgraphStructureEncoder,
    CombinedPositionalEncoder
)

__all__ = [
    # 列编码器
    'NumericalEncoder',
    'CategoricalEncoder',
    'TextEncoder',
    'TimeEncoder',
    'EmbeddingEncoder',
    'MultiModalColumnEncoder',

    # 表编码器
    'TableEncoder',
    'TableTransformer',
    'RowAggregator',

    # 位置编码器
    'PositionalEncoding',
    'NodeTypeEncoder',
    'HopEncoder',
    'TemporalEncoder',
    'SubgraphStructureEncoder',
    'CombinedPositionalEncoder'
]

# 数据类型到编码器的映射
DTYPE_TO_ENCODER = {
    'numerical': NumericalEncoder,
    'categorical': CategoricalEncoder,
    'text': TextEncoder,
    'timestamp': TimeEncoder,
    'embedding': EmbeddingEncoder
}