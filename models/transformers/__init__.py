"""
Transformer模块
包含注意力机制和图Transformer
"""

from .attention import (
    MultiHeadAttention,
    GraphAttention,
    CrossAttention,
    SelfAttentionBlock
)

from .relational_graph_transformer import (
    RelationalGraphTransformer,
    RelationalGraphTransformerLayer,
    HeterogeneousRelationalGraphTransformer
)

__all__ = [
    # 注意力机制
    'MultiHeadAttention',
    'GraphAttention',
    'CrossAttention',
    'SelfAttentionBlock',

    # 图Transformer
    'RelationalGraphTransformer',
    'RelationalGraphTransformerLayer',
    'HeterogeneousRelationalGraphTransformer'
]

# Transformer配置默认值
DEFAULT_TRANSFORMER_CONFIG = {
    'num_heads': 8,
    'num_layers': 6,
    'hidden_dim': 768,
    'feedforward_dim': 3072,
    'dropout': 0.1,
    'attention_dropout': 0.1,
    'activation': 'gelu'
}