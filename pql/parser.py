"""
PQL (Predictive Query Language) 解析器
解析预测查询语言，生成任务配置
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import ast


@dataclass
class PQLQuery:
    """PQL查询的结构化表示"""

    # PREDICT子句
    aggregation: str  # COUNT, SUM, AVG, EXISTS, FIRST, LIST_DISTINCT等
    target_column: Optional[str]  # 目标列
    target_table: Optional[str]  # 目标表
    time_window_start: int  # 时间窗口开始（相对天数）
    time_window_end: int  # 时间窗口结束（相对天数）

    # FOR子句
    entity_table: str  # 实体表
    entity_column: str  # 实体列（主键）
    entity_values: List[Any]  # 实体值列表

    # WHERE子句（可选）
    where_conditions: Optional[Dict[str, Any]] = None

    # 任务类型（推断得出）
    task_type: Optional[str] = None

    # 原始查询
    raw_query: Optional[str] = None


class PQLParser:
    """
    PQL解析器

    支持的语法示例：
    - PREDICT COUNT(orders.*, 0, 7) FOR users.user_id = 123
    - PREDICT SUM(orders.value, 0, 30) FOR users.user_id IN (1, 2, 3)
    - PREDICT EXISTS(reviews.*, 0, 1) > 0 FOR products.product_id = 456
    - PREDICT LIST_DISTINCT(orders.item_id, 0, 7) FOR users.user_id = 789
    """

    def __init__(self):
        # 聚合函数到任务类型的映射
        self.aggregation_to_task = {
            'COUNT': 'regression',
            'SUM': 'regression',
            'AVG': 'regression',
            'EXISTS': 'classification',
            'FIRST': 'classification',
            'LIST_DISTINCT': 'link_prediction',
            'MAX': 'regression',
            'MIN': 'regression'
        }

        # 编译正则表达式
        self._compile_patterns()

    def _compile_patterns(self):
        """编译正则表达式模式"""
        # PREDICT子句模式
        self.predict_pattern = re.compile(
            r'PREDICT\s+(\w+)\s*\(([\w\.\*]+)(?:\s*,\s*(-?\d+)\s*,\s*(-?\d+))?\)',
            re.IGNORECASE
        )

        # 比较运算符模式（用于二分类）
        self.comparison_pattern = re.compile(
            r'PREDICT\s+.*?\s*([><=]+)\s*(\d+)',
            re.IGNORECASE
        )

        # FOR子句模式
        self.for_pattern = re.compile(
            r'FOR\s+(?:EACH\s+)?([\w\.]+)\s*(?:=\s*(.+?)|IN\s*\((.+?)\))',
            re.IGNORECASE
        )

        # WHERE子句模式
        self.where_pattern = re.compile(
            r'WHERE\s+(.+)$',
            re.IGNORECASE
        )

    def parse(self, query: str) -> PQLQuery:
        """
        解析PQL查询

        query: PQL查询字符串
        返回: PQLQuery对象
        """
        query = query.strip()

        # 1. 解析PREDICT子句
        predict_match = self.predict_pattern.search(query)
        if not predict_match:
            raise ValueError(f"无效的PREDICT子句: {query}")

        aggregation = predict_match.group(1).upper()
        target = predict_match.group(2)

        # 解析时间窗口
        if predict_match.group(3) and predict_match.group(4):
            time_start = int(predict_match.group(3))
            time_end = int(predict_match.group(4))
        else:
            # 默认时间窗口
            time_start = 0
            time_end = 7

        # 解析目标表和列
        if '.' in target:
            target_table, target_column = target.split('.', 1)
        else:
            target_table = None
            target_column = target

        # 2. 检查是否有比较运算符（二分类）
        task_type = self.aggregation_to_task.get(aggregation)
        comparison_match = self.comparison_pattern.search(query)
        if comparison_match:
            task_type = 'classification'

        # 3. 解析FOR子句
        for_match = self.for_pattern.search(query)
        if not for_match:
            raise ValueError(f"无效的FOR子句: {query}")

        entity_spec = for_match.group(1)
        if '.' in entity_spec:
            entity_table, entity_column = entity_spec.split('.', 1)
        else:
            raise ValueError(f"FOR子句必须指定表和列: {entity_spec}")

        # 解析实体值
        if for_match.group(2):  # 单个值
            entity_values = [self._parse_value(for_match.group(2))]
        elif for_match.group(3):  # IN子句
            values_str = for_match.group(3)
            entity_values = [
                self._parse_value(v.strip())
                for v in values_str.split(',')
            ]
        else:
            entity_values = []

        # 4. 解析WHERE子句（可选）
        where_conditions = None
        where_match = self.where_pattern.search(query)
        if where_match:
            where_conditions = self._parse_where_clause(where_match.group(1))

        # 创建查询对象
        pql_query = PQLQuery(
            aggregation=aggregation,
            target_column=target_column,
            target_table=target_table,
            time_window_start=time_start,
            time_window_end=time_end,
            entity_table=entity_table,
            entity_column=entity_column,
            entity_values=entity_values,
            where_conditions=where_conditions,
            task_type=task_type,
            raw_query=query
        )

        return pql_query

    def _parse_value(self, value_str: str) -> Any:
        """解析值字符串"""
        value_str = value_str.strip()

        # 尝试解析为数字
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        # 尝试解析为布尔值
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'

        # 去除引号
        if (value_str.startswith('"') and value_str.endswith('"')) or \
                (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]

        return value_str

    def _parse_where_clause(self, where_str: str) -> Dict[str, Any]:
        """
        解析WHERE子句
        简化实现，仅支持AND连接的简单条件
        """
        conditions = {}

        # 分割AND条件
        and_parts = re.split(r'\s+AND\s+', where_str, flags=re.IGNORECASE)

        for part in and_parts:
            # 解析单个条件
            match = re.match(r'([\w\.]+)\s*([><=]+)\s*(.+)', part.strip())
            if match:
                field = match.group(1)
                operator = match.group(2)
                value = self._parse_value(match.group(3))

                conditions[field] = {
                    'operator': operator,
                    'value': value
                }

        return conditions

    def to_task_config(self, pql_query: PQLQuery) -> Dict[str, Any]:
        """
        将PQL查询转换为任务配置

        返回: 任务配置字典
        """
        config = {
            'task_type': pql_query.task_type,
            'aggregation': pql_query.aggregation,
            'target_table': pql_query.target_table,
            'target_column': pql_query.target_column,
            'time_window': {
                'start': pql_query.time_window_start,
                'end': pql_query.time_window_end
            },
            'entity': {
                'table': pql_query.entity_table,
                'column': pql_query.entity_column,
                'values': pql_query.entity_values
            }
        }

        # 添加WHERE条件
        if pql_query.where_conditions:
            config['filters'] = pql_query.where_conditions

        # 特定任务类型的配置
        if pql_query.task_type == 'classification':
            # 推断类别数
            if pql_query.aggregation == 'EXISTS':
                config['num_classes'] = 2  # 二分类
            elif pql_query.aggregation == 'FIRST':
                # 需要从数据中推断
                config['num_classes'] = None

        elif pql_query.task_type == 'link_prediction':
            # 链接预测特定配置
            if pql_query.aggregation == 'LIST_DISTINCT':
                config['prediction_type'] = 'recommendation'
                config['max_predictions'] = 10  # 默认返回10个

        return config

    def validate(self, pql_query: PQLQuery, database_schema: Dict[str, List[str]]) -> List[str]:
        """
        验证PQL查询的有效性

        database_schema: {表名: [列名列表]}
        返回: 错误消息列表
        """
        errors = []

        # 验证实体表
        if pql_query.entity_table not in database_schema:
            errors.append(f"实体表 '{pql_query.entity_table}' 不存在")
        elif pql_query.entity_column not in database_schema[pql_query.entity_table]:
            errors.append(
                f"列 '{pql_query.entity_column}' 在表 '{pql_query.entity_table}' 中不存在"
            )

        # 验证目标表
        if pql_query.target_table:
            if pql_query.target_table not in database_schema:
                errors.append(f"目标表 '{pql_query.target_table}' 不存在")
            elif pql_query.target_column != '*' and \
                    pql_query.target_column not in database_schema[pql_query.target_table]:
                errors.append(
                    f"列 '{pql_query.target_column}' 在表 '{pql_query.target_table}' 中不存在"
                )

        # 验证时间窗口
        if pql_query.time_window_start > pql_query.time_window_end:
            errors.append(
                f"无效的时间窗口: 开始时间 {pql_query.time_window_start} > " +
                f"结束时间 {pql_query.time_window_end}"
            )

        # 验证WHERE条件中的字段
        if pql_query.where_conditions:
            for field in pql_query.where_conditions:
                if '.' in field:
                    table, column = field.split('.', 1)
                    if table not in database_schema:
                        errors.append(f"WHERE子句中的表 '{table}' 不存在")
                    elif column not in database_schema[table]:
                        errors.append(f"WHERE子句中的列 '{column}' 在表 '{table}' 中不存在")

        return errors


class PQLExecutor:
    """
    PQL执行器
    执行解析后的PQL查询
    """

    def __init__(self, database, model):
        self.database = database
        self.model = model
        self.parser = PQLParser()

    def execute(self, query: str, prediction_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        执行PQL查询

        query: PQL查询字符串
        prediction_time: 预测时间（默认为当前时间）

        返回: 预测结果
        """
        # 解析查询
        pql_query = self.parser.parse(query)

        # 验证查询
        schema = self._get_database_schema()
        errors = self.parser.validate(pql_query, schema)
        if errors:
            raise ValueError(f"查询验证失败: {'; '.join(errors)}")

        # 转换为任务配置
        task_config = self.parser.to_task_config(pql_query)

        # 设置预测时间
        if prediction_time is None:
            prediction_time = datetime.now()

        # 执行预测
        results = {}
        for entity_value in pql_query.entity_values:
            # 获取实体ID
            entity_id = self._get_entity_id(
                pql_query.entity_table,
                pql_query.entity_column,
                entity_value
            )

            if entity_id is not None:
                # 执行预测
                prediction = self.model.predict(
                    database=self.database,
                    entity_id=entity_id,
                    prediction_time=prediction_time,
                    task_config=task_config
                )

                results[entity_value] = prediction

        return results

    def _get_database_schema(self) -> Dict[str, List[str]]:
        """获取数据库模式"""
        schema = {}
        for table_name, table in self.database.tables.items():
            schema[table_name] = [col.name for col in table.columns]
        return schema

    def _get_entity_id(self, table: str, column: str, value: Any) -> Optional[int]:
        """获取实体的图节点ID"""
        # 这里需要实现从数据库值到图节点ID的映射
        # 简化实现
        return hash((table, column, value)) % 1000000