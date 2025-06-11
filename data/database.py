"""
关系型数据库接口
用于管理表、主外键关系和数据访问
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class Column:
    """列的元数据"""
    name: str
    dtype: str  # numerical, categorical, text, timestamp, embedding
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_table: Optional[str] = None
    foreign_column: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Table:
    """表的元数据和数据"""
    name: str
    columns: List[Column]
    data: pd.DataFrame
    primary_key: Optional[str] = None

    def __post_init__(self):
        """验证表结构"""
        column_names = [col.name for col in self.columns]
        assert set(column_names) == set(self.data.columns), \
            "列定义与数据框列不匹配"

        # 设置主键
        for col in self.columns:
            if col.is_primary_key:
                self.primary_key = col.name
                break


class RelationalDatabase:
    """
    关系型数据库的抽象表示
    管理多个表及其关系
    """

    def __init__(self):
        self.tables: Dict[str, Table] = {}
        self.relationships: List[Tuple[str, str, str, str]] = []  # (表1, 列1, 表2, 列2)

    def add_table(self, table: Table):
        """添加表到数据库"""
        if table.name in self.tables:
            raise ValueError(f"表 {table.name} 已存在")

        self.tables[table.name] = table

        # 自动检测外键关系
        for col in table.columns:
            if col.is_foreign_key and col.foreign_table and col.foreign_column:
                self.add_relationship(
                    table.name, col.name,
                    col.foreign_table, col.foreign_column
                )

    def add_relationship(self, table1: str, column1: str,
                         table2: str, column2: str):
        """添加表之间的关系（主外键）"""
        # 验证表和列存在
        if table1 not in self.tables:
            raise ValueError(f"表 {table1} 不存在")
        if table2 not in self.tables:
            raise ValueError(f"表 {table2} 不存在")

        # 验证列存在
        if column1 not in self.tables[table1].data.columns:
            raise ValueError(f"列 {column1} 在表 {table1} 中不存在")
        if column2 not in self.tables[table2].data.columns:
            raise ValueError(f"列 {column2} 在表 {table2} 中不存在")

        self.relationships.append((table1, column1, table2, column2))

    def get_table(self, table_name: str) -> Table:
        """获取表"""
        if table_name not in self.tables:
            raise ValueError(f"表 {table_name} 不存在")
        return self.tables[table_name]

    def get_related_tables(self, table_name: str) -> List[str]:
        """获取与指定表相关的所有表"""
        related = set()
        for t1, c1, t2, c2 in self.relationships:
            if t1 == table_name:
                related.add(t2)
            elif t2 == table_name:
                related.add(t1)
        return list(related)

    def get_join_path(self, table1: str, table2: str) -> Optional[List[Tuple[str, str, str, str]]]:
        """
        获取两个表之间的连接路径
        返回连接关系列表，如果不存在路径则返回None
        """
        # 简单的BFS查找路径
        from collections import deque

        if table1 == table2:
            return []

        visited = {table1}
        queue = deque([(table1, [])])

        while queue:
            current_table, path = queue.popleft()

            # 查找所有相邻表
            for t1, c1, t2, c2 in self.relationships:
                next_table = None
                if t1 == current_table and t2 not in visited:
                    next_table = t2
                    new_rel = (t1, c1, t2, c2)
                elif t2 == current_table and t1 not in visited:
                    next_table = t1
                    new_rel = (t2, c2, t1, c1)

                if next_table:
                    new_path = path + [new_rel]
                    if next_table == table2:
                        return new_path

                    visited.add(next_table)
                    queue.append((next_table, new_path))

        return None

    def execute_join(self, tables: List[str],
                     conditions: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        执行多表连接
        conditions: 过滤条件字典
        """
        if not tables:
            raise ValueError("至少需要一个表")

        result = self.tables[tables[0]].data.copy()

        for i in range(1, len(tables)):
            # 找到连接路径
            join_path = self.get_join_path(tables[i - 1], tables[i])
            if not join_path:
                raise ValueError(f"表 {tables[i - 1]} 和 {tables[i]} 之间没有连接路径")

            # 执行连接
            for t1, c1, t2, c2 in join_path:
                if t1 in result.columns or t1 == tables[i - 1]:
                    result = pd.merge(
                        result,
                        self.tables[t2].data,
                        left_on=c1,
                        right_on=c2,
                        how='inner'
                    )

        # 应用过滤条件
        if conditions:
            for col, value in conditions.items():
                if col in result.columns:
                    result = result[result[col] == value]

        return result

    def get_column_statistics(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """获取列的统计信息"""
        table = self.get_table(table_name)
        data = table.data[column_name]

        stats = {
            'count': len(data),
            'null_count': data.isnull().sum(),
            'unique_count': data.nunique()
        }

        # 根据列类型添加特定统计信息
        col_meta = next((c for c in table.columns if c.name == column_name), None)
        if col_meta:
            if col_meta.dtype == 'numerical':
                stats.update({
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max()
                })
            elif col_meta.dtype == 'categorical':
                stats['value_counts'] = data.value_counts().to_dict()

        return stats