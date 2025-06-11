"""
KumoRFM - 关系型数据的基础模型

用于在关系型数据上进行上下文学习的预训练模型
"""

__version__ = "0.1.0"


# 导入主要组件
from .config.model_config import KumoRFMConfig
from .models.kumo_rfm import KumoRFM
from .data.database import RelationalDatabase, Table, Column
from .data.graph_builder import GraphBuilder, HeterogeneousGraph
from .pql.parser import PQLParser, PQLExecutor, PQLQuery
from .evaluation.evaluator import Evaluator
from .evaluation.metrics import Metrics

# 导入工具函数
from .utils.data_utils import (
    load_csv_to_database,
    infer_column_type,
    split_temporal_data,
    create_temporal_features,
    handle_missing_values
)

from .utils.training_utils import (
    set_random_seed,
    EarlyStopping,
    ModelCheckpoint,
    MetricTracker
)

# 定义公开API
__all__ = [
    # 版本信息
    "__version__",

    # 核心类
    "KumoRFM",
    "KumoRFMConfig",

    # 数据结构
    "RelationalDatabase",
    "Table",
    "Column",
    "GraphBuilder",
    "HeterogeneousGraph",

    # PQL
    "PQLParser",
    "PQLExecutor",
    "PQLQuery",

    # 评估
    "Evaluator",
    "Metrics",

    # 工具函数
    "load_csv_to_database",
    "infer_column_type",
    "split_temporal_data",
    "create_temporal_features",
    "handle_missing_values",
    "set_random_seed",
    "EarlyStopping",
    "ModelCheckpoint",
    "MetricTracker",
]


# 打印欢迎信息
def _print_welcome():
    """打印欢迎信息"""
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                     KumoRFM v{__version__}                ║
║        关系型数据的基础模型 - 上下文学习框架                     ║
╚═══════════════════════════════════════════════════════════╝
    """)

# 可选：在导入时打印欢迎信息
# _print_welcome()