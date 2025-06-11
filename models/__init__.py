"""
模型模块
包含KumoRFM模型及其组件
"""

from .kumo_rfm import KumoRFM

# 导入子模块
from . import encoders
from . import transformers
from . import icl

__all__ = [
    'KumoRFM',
    'encoders',
    'transformers',
    'icl'
]

# 模型版本
MODEL_VERSION = "1.0.0"

# 支持的任务类型
SUPPORTED_TASKS = [
    'classification',
    'regression',
    'link_prediction',
    'exists',
    'count',
    'sum',
    'next'
]

def is_supported_task(task_type: str) -> bool:
    """检查任务类型是否支持"""
    return task_type.lower() in SUPPORTED_TASKS