"""
数据处理工具函数
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import pickle
import json

from data.database import RelationalDatabase, Table, Column

logger = logging.getLogger(__name__)


def load_csv_to_database(csv_files: Dict[str, str],
                         relationships: List[Tuple[str, str, str, str]],
                         column_types: Optional[Dict[str, Dict[str, str]]] = None) -> RelationalDatabase:
    """
    从CSV文件加载关系型数据库

    csv_files: {表名: CSV文件路径}
    relationships: [(表1, 列1, 表2, 列2), ...]
    column_types: {表名: {列名: 类型}}

    返回: RelationalDatabase对象
    """
    database = RelationalDatabase()

    for table_name, csv_path in csv_files.items():
        logger.info(f"加载表 {table_name} 从 {csv_path}")

        # 读取CSV
        df = pd.read_csv(csv_path)

        # 推断列类型
        columns = []
        for col_name in df.columns:
            # 获取指定的列类型或推断
            if column_types and table_name in column_types and col_name in column_types[table_name]:
                dtype = column_types[table_name][col_name]
            else:
                dtype = infer_column_type(df[col_name])

            column = Column(
                name=col_name,
                dtype=dtype,
                is_primary_key=(col_name.endswith('_id') and col_name == f"{table_name[:-1]}_id")
            )
            columns.append(column)

        # 创建表
        table = Table(
            name=table_name,
            columns=columns,
            data=df
        )

        database.add_table(table)

    # 添加关系
    for rel in relationships:
        database.add_relationship(*rel)

    logger.info(f"数据库加载完成: {len(database.tables)} 个表")

    return database


def infer_column_type(series: pd.Series) -> str:
    """
    推断列的数据类型

    返回: 'numerical', 'categorical', 'text', 'timestamp', 'embedding'
    """
    # 检查是否为时间戳
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'timestamp'

    # 尝试转换为时间戳
    try:
        pd.to_datetime(series.dropna().iloc[:100])
        return 'timestamp'
    except:
        pass

    # 检查是否为数值
    if pd.api.types.is_numeric_dtype(series):
        return 'numerical'

    # 检查唯一值数量
    unique_ratio = series.nunique() / len(series)

    # 如果唯一值比例很高，可能是文本
    if unique_ratio > 0.5:
        # 检查平均长度
        avg_length = series.dropna().astype(str).str.len().mean()
        if avg_length > 50:
            return 'text'

    # 否则视为分类变量
    return 'categorical'


def split_temporal_data(data: pd.DataFrame,
                        timestamp_column: str,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序划分数据集

    data: 数据框
    timestamp_column: 时间戳列名
    train_ratio: 训练集比例
    val_ratio: 验证集比例
    test_ratio: 测试集比例

    返回: (训练集, 验证集, 测试集)
    """
    # 按时间排序
    data_sorted = data.sort_values(timestamp_column)

    n = len(data_sorted)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_data = data_sorted.iloc[:train_size]
    val_data = data_sorted.iloc[train_size:train_size + val_size]
    test_data = data_sorted.iloc[train_size + val_size:]

    logger.info(f"数据集划分: 训练集 {len(train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")

    return train_data, val_data, test_data


def create_temporal_features(df: pd.DataFrame,
                             timestamp_column: str,
                             feature_prefix: str = 'time') -> pd.DataFrame:
    """
    创建时间特征

    df: 数据框
    timestamp_column: 时间戳列名
    feature_prefix: 特征前缀

    返回: 添加了时间特征的数据框
    """
    df = df.copy()

    # 转换为datetime
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # 提取时间特征
    df[f'{feature_prefix}_year'] = df[timestamp_column].dt.year
    df[f'{feature_prefix}_month'] = df[timestamp_column].dt.month
    df[f'{feature_prefix}_day'] = df[timestamp_column].dt.day
    df[f'{feature_prefix}_dayofweek'] = df[timestamp_column].dt.dayofweek
    df[f'{feature_prefix}_hour'] = df[timestamp_column].dt.hour
    df[f'{feature_prefix}_quarter'] = df[timestamp_column].dt.quarter
    df[f'{feature_prefix}_dayofyear'] = df[timestamp_column].dt.dayofyear
    df[f'{feature_prefix}_weekofyear'] = df[timestamp_column].dt.isocalendar().week

    # 周期性编码
    df[f'{feature_prefix}_month_sin'] = np.sin(2 * np.pi * df[f'{feature_prefix}_month'] / 12)
    df[f'{feature_prefix}_month_cos'] = np.cos(2 * np.pi * df[f'{feature_prefix}_month'] / 12)

    df[f'{feature_prefix}_day_sin'] = np.sin(2 * np.pi * df[f'{feature_prefix}_day'] / 31)
    df[f'{feature_prefix}_day_cos'] = np.cos(2 * np.pi * df[f'{feature_prefix}_day'] / 31)

    df[f'{feature_prefix}_hour_sin'] = np.sin(2 * np.pi * df[f'{feature_prefix}_hour'] / 24)
    df[f'{feature_prefix}_hour_cos'] = np.cos(2 * np.pi * df[f'{feature_prefix}_hour'] / 24)

    return df


def handle_missing_values(df: pd.DataFrame,
                          strategy: str = 'auto',
                          numeric_fill: Union[str, float] = 'mean',
                          categorical_fill: Union[str, Any] = 'mode') -> pd.DataFrame:
    """
    处理缺失值

    df: 数据框
    strategy: 处理策略 ('auto', 'drop', 'fill')
    numeric_fill: 数值列填充策略 ('mean', 'median', 'zero', 或具体值)
    categorical_fill: 分类列填充策略 ('mode', 'unknown', 或具体值)

    返回: 处理后的数据框
    """
    df = df.copy()

    if strategy == 'drop':
        return df.dropna()

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                # 数值列
                if numeric_fill == 'mean':
                    fill_value = df[col].mean()
                elif numeric_fill == 'median':
                    fill_value = df[col].median()
                elif numeric_fill == 'zero':
                    fill_value = 0
                else:
                    fill_value = numeric_fill

                df[col].fillna(fill_value, inplace=True)

            else:
                # 分类列
                if categorical_fill == 'mode':
                    mode_value = df[col].mode()
                    fill_value = mode_value[0] if len(mode_value) > 0 else 'unknown'
                elif categorical_fill == 'unknown':
                    fill_value = 'unknown'
                else:
                    fill_value = categorical_fill

                df[col].fillna(fill_value, inplace=True)

    return df


def encode_categorical_features(df: pd.DataFrame,
                                categorical_columns: List[str],
                                encoding_type: str = 'label',
                                min_frequency: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    编码分类特征

    df: 数据框
    categorical_columns: 分类列名列表
    encoding_type: 编码类型 ('label', 'onehot', 'frequency')
    min_frequency: 最小频率（用于低频类别处理）

    返回: (编码后的数据框, 编码器字典)
    """
    df = df.copy()
    encoders = {}

    for col in categorical_columns:
        if col not in df.columns:
            continue

        if encoding_type == 'label':
            # 标签编码
            # 处理低频类别
            value_counts = df[col].value_counts()
            low_freq_values = value_counts[value_counts < min_frequency].index
            df.loc[df[col].isin(low_freq_values), col] = 'OTHER'

            # 创建映射
            unique_values = df[col].unique()
            value_to_id = {val: idx for idx, val in enumerate(unique_values)}

            # 应用编码
            df[col] = df[col].map(value_to_id)

            encoders[col] = {
                'type': 'label',
                'mapping': value_to_id,
                'inverse_mapping': {idx: val for val, idx in value_to_id.items()}
            }

        elif encoding_type == 'frequency':
            # 频率编码
            freq_encoding = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq_encoding)

            encoders[col] = {
                'type': 'frequency',
                'mapping': freq_encoding
            }

    return df, encoders


def normalize_numerical_features(df: pd.DataFrame,
                                 numerical_columns: List[str],
                                 method: str = 'standard') -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    归一化数值特征

    df: 数据框
    numerical_columns: 数值列名列表
    method: 归一化方法 ('standard', 'minmax', 'robust')

    返回: (归一化后的数据框, 归一化参数)
    """
    df = df.copy()
    normalizers = {}

    for col in numerical_columns:
        if col not in df.columns:
            continue

        if method == 'standard':
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / (std + 1e-8)

            normalizers[col] = {'mean': mean, 'std': std}

        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)

            normalizers[col] = {'min': min_val, 'max': max_val}

        elif method == 'robust':
            median = df[col].median()
            mad = (df[col] - median).abs().median()
            df[col] = (df[col] - median) / (mad + 1e-8)

            normalizers[col] = {'median': median, 'mad': mad}

    return df, normalizers


def create_graph_features(database: RelationalDatabase,
                          entity_table: str,
                          max_depth: int = 2) -> pd.DataFrame:
    """
    创建基于图结构的特征

    database: 数据库对象
    entity_table: 实体表名
    max_depth: 最大深度

    返回: 图特征数据框
    """
    features = []

    entity_data = database.get_table(entity_table).data

    for idx, row in entity_data.iterrows():
        entity_features = {
            'entity_id': row[database.tables[entity_table].primary_key]
        }

        # 计算不同深度的邻居数
        for depth in range(1, max_depth + 1):
            # 这里简化处理，实际应该通过图遍历计算
            related_tables = database.get_related_tables(entity_table)

            for related_table in related_tables:
                feature_name = f'num_{related_table}_depth_{depth}'
                # 简化：使用随机数模拟
                entity_features[feature_name] = np.random.randint(0, 10)

        features.append(entity_features)

    return pd.DataFrame(features)


def save_processed_data(data: Any, filepath: str, format: str = 'pickle'):
    """
    保存处理后的数据

    data: 要保存的数据
    filepath: 文件路径
    format: 保存格式 ('pickle', 'json', 'csv')
    """
    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    elif format == 'json':
        with open(filepath, 'w') as f:
            json.dump(data, f, default=str)

    elif format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            raise ValueError("CSV格式只支持DataFrame")

    else:
        raise ValueError(f"不支持的格式: {format}")

    logger.info(f"数据已保存到: {filepath}")


def load_processed_data(filepath: str, format: str = 'pickle') -> Any:
    """
    加载处理后的数据

    filepath: 文件路径
    format: 文件格式

    返回: 加载的数据
    """
    if format == 'pickle':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

    elif format == 'json':
        with open(filepath, 'r') as f:
            data = json.load(f)

    elif format == 'csv':
        data = pd.read_csv(filepath)

    else:
        raise ValueError(f"不支持的格式: {format}")

    logger.info(f"数据已从 {filepath} 加载")

    return data


def create_dataset_statistics(database: RelationalDatabase) -> Dict[str, Any]:
    """
    创建数据集统计信息

    database: 数据库对象

    返回: 统计信息字典
    """
    stats = {
        'num_tables': len(database.tables),
        'num_relationships': len(database.relationships),
        'tables': {}
    }

    for table_name, table in database.tables.items():
        table_stats = {
            'num_rows': len(table.data),
            'num_columns': len(table.columns),
            'columns': {}
        }

        for col in table.columns:
            col_data = table.data[col.name]

            col_stats = {
                'dtype': col.dtype,
                'null_count': col_data.isnull().sum(),
                'null_ratio': col_data.isnull().mean()
            }

            if col.dtype == 'numerical':
                col_stats.update({
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'median': col_data.median()
                })

            elif col.dtype == 'categorical':
                col_stats.update({
                    'unique_values': col_data.nunique(),
                    'top_values': col_data.value_counts().head(10).to_dict()
                })

            table_stats['columns'][col.name] = col_stats

        stats['tables'][table_name] = table_stats

    return stats