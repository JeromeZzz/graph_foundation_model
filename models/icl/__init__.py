"""
上下文学习（In-Context Learning）模块
实现KumoRFM的核心ICL功能
"""

from .icl_module import (
    InContextLearningModule,
    ICLLayer,
    LabelEncoder,
    ClassificationHead,
    RegressionHead,
    LinkPredictionHead
)

from .context_generator import ContextGenerator

__all__ = [
    # ICL模块
    'InContextLearningModule',
    'ICLLayer',
    'LabelEncoder',

    # 任务头
    'ClassificationHead',
    'RegressionHead',
    'LinkPredictionHead',

    # 上下文生成
    'ContextGenerator'
]

# ICL配置默认值
DEFAULT_ICL_CONFIG = {
    'num_context_examples': 32,
    'context_aggregation': 'attention',
    'use_prototypes': True,
    'use_context_stats': True
}