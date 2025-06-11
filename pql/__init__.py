"""
PQL (Predictive Query Language) 模块
提供预测查询语言的解析和执行功能
"""

from .parser import PQLParser, PQLExecutor
from .query import (
    PQLQuery,
    PQLQueryBuilder,
    PredictClause,
    ForClause,
    WhereClause,
    Condition,
    ColumnReference,
    TimeWindow,
    AggregationType,
    OperatorType
)
from .validator import PQLValidator, ValidationError

__all__ = [
    # 解析器和执行器
    'PQLParser',
    'PQLExecutor',

    # 查询对象
    'PQLQuery',
    'PQLQueryBuilder',
    'PredictClause',
    'ForClause',
    'WhereClause',
    'Condition',
    'ColumnReference',
    'TimeWindow',

    # 枚举类型
    'AggregationType',
    'OperatorType',

    # 验证器
    'PQLValidator',
    'ValidationError'
]

# PQL示例
EXAMPLE_QUERIES = {
    'churn_prediction': "PREDICT COUNT(orders.*, 0, 30) > 0 FOR users.user_id = 123",
    'revenue_forecast': "PREDICT SUM(orders.value, 0, 7) FOR users.user_id IN (1, 2, 3)",
    'recommendation': "PREDICT LIST_DISTINCT(orders.item_id, 0, 7) FOR users.user_id = 456",
    'fraud_detection': "PREDICT EXISTS(transactions.*, 0, 1) > 0 FOR accounts.account_id = 789"
}


def parse_query(query_string: str) -> PQLQuery:
    """快捷解析函数"""
    parser = PQLParser()
    return parser.parse(query_string)