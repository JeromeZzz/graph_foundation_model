"""
评估模块
提供模型评估和指标计算功能
"""

from .evaluator import Evaluator
from .metrics import Metrics

__all__ = [
    'Evaluator',
    'Metrics'
]

# 指标别名
METRIC_ALIASES = {
    'auc': 'auroc',
    'ap': 'auprc',
    'acc': 'accuracy',
    'prec': 'precision',
    'rec': 'recall',
    'map': 'map@k'
}

def get_metric_name(alias: str) -> str:
    """获取标准指标名称"""
    return METRIC_ALIASES.get(alias.lower(), alias.lower())