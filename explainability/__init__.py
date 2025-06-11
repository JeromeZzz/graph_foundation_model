"""
可解释性模块
提供模型预测的解释功能
"""

from .analytical_explain import AnalyticalExplainer
from .gradient_explain import GradientExplainer, ExplainabilityModule

__all__ = [
    'AnalyticalExplainer',
    'GradientExplainer',
    'ExplainabilityModule'
]

# 解释方法类型
EXPLANATION_METHODS = {
    'analytical': AnalyticalExplainer,
    'gradient': GradientExplainer,
    'combined': ExplainabilityModule
}


def create_explainer(method: str, model=None):
    """
    创建解释器

    method: 解释方法 ('analytical', 'gradient', 'combined')
    model: 模型对象（gradient和combined方法需要）
    """
    if method not in EXPLANATION_METHODS:
        raise ValueError(f"不支持的解释方法: {method}")

    explainer_class = EXPLANATION_METHODS[method]

    if method == 'analytical':
        return explainer_class()
    elif method in ['gradient', 'combined']:
        if model is None:
            raise ValueError(f"{method} 方法需要提供模型")
        return explainer_class(model)