"""
工具函数模块
提供各种辅助功能
"""

from .data_utils import (
    load_csv_to_database,
    infer_column_type,
    split_temporal_data,
    create_temporal_features,
    handle_missing_values,
    encode_categorical_features,
    normalize_numerical_features,
    create_graph_features,
    save_processed_data,
    load_processed_data,
    create_dataset_statistics
)

from .training_utils import (
    set_random_seed,
    EarlyStopping,
    ModelCheckpoint,
    GradientAccumulator,
    MetricTracker,
    count_parameters,
    save_config,
    create_optimizer,
    create_scheduler
)

__all__ = [
    # 数据工具
    'load_csv_to_database',
    'infer_column_type',
    'split_temporal_data',
    'create_temporal_features',
    'handle_missing_values',
    'encode_categorical_features',
    'normalize_numerical_features',
    'create_graph_features',
    'save_processed_data',
    'load_processed_data',
    'create_dataset_statistics',

    # 训练工具
    'set_random_seed',
    'EarlyStopping',
    'ModelCheckpoint',
    'GradientAccumulator',
    'MetricTracker',
    'count_parameters',
    'save_config',
    'create_optimizer',
    'create_scheduler'
]

# 工具函数版本
UTILS_VERSION = '0.1.0'

# 默认随机种子
DEFAULT_SEED = 42

# 设置默认随机种子
set_random_seed(DEFAULT_SEED)

import logging


# 配置日志
def setup_logging(level=logging.INFO, log_file=None):
    """
    设置日志配置

    level: 日志级别
    log_file: 日志文件路径（可选）
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # 设置特定模块的日志级别
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)


# 默认设置日志
setup_logging()