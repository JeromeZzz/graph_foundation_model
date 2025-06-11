"""
配置模块
提供模型和系统配置
"""

from .model_config import (
    KumoRFMConfig,
    ColumnEncoderConfig,
    SamplerConfig
)

__all__ = [
    'KumoRFMConfig',
    'ColumnEncoderConfig',
    'SamplerConfig'
]

# 默认配置
DEFAULT_CONFIG = KumoRFMConfig()

def load_config(config_path: str = None) -> KumoRFMConfig:
    """
    加载配置文件

    config_path: 配置文件路径（YAML或JSON）
    """
    if config_path is None:
        return DEFAULT_CONFIG

    import yaml
    import json

    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path}")

    return KumoRFMConfig(**config_dict)