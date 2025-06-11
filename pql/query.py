"""
PQL查询对象和构建器
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum


class AggregationType(Enum):
    """聚合类型枚举"""
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    FIRST = "FIRST"
    LAST = "LAST"
    EXISTS = "EXISTS"
    LIST_DISTINCT = "LIST_DISTINCT"


class OperatorType(Enum):
    """操作符类型枚举"""
    EQ = "="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    IN = "IN"
    NOT_IN = "NOT IN"
    CONTAINS = "CONTAINS"
    NOT_CONTAINS = "NOT CONTAINS"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


@dataclass
class TimeWindow:
    """时间窗口"""
    start_offset: int  # 开始偏移（天数）
    end_offset: int    # 结束偏移（天数）
    unit: str = "days"  # 时间单位

    def to_timedelta(self, offset: int) -> timedelta:
        """转换为timedelta对象"""
        if self.unit == "days":
            return timedelta(days=offset)
        elif self.unit == "hours":
            return timedelta(hours=offset)
        elif self.unit == "minutes":
            return timedelta(minutes=offset)
        else:
            raise ValueError(f"不支持的时间单位: {self.unit}")


@dataclass
class ColumnReference:
    """列引用"""
    table: Optional[str]
    column: str

    def __str__(self):
        if self.table:
            return f"{self.table}.{self.column}"
        return self.column

    @classmethod
    def parse(cls, ref_str: str) -> 'ColumnReference':
        """解析列引用字符串"""
        parts = ref_str.split('.')
        if len(parts) == 2:
            return cls(table=parts[0], column=parts[1])
        elif len(parts) == 1:
            return cls(table=None, column=parts[0])
        else:
            raise ValueError(f"无效的列引用: {ref_str}")


@dataclass
class Condition:
    """查询条件"""
    column: ColumnReference
    operator: OperatorType
    value: Any

    def evaluate(self, row_data: Dict[str, Any]) -> bool:
        """评估条件是否满足"""
        col_value = row_data.get(str(self.column))
        
        if self.operator == OperatorType.EQ:
            return col_value == self.value
        elif self.operator == OperatorType.NE:
            return col_value != self.value
        elif self.operator == OperatorType.GT:
            return col_value > self.value
        elif self.operator == OperatorType.GE:
            return col_value >= self.value
        elif self.operator == OperatorType.LT:
            return col_value < self.value
        elif self.operator == OperatorType.LE:
            return col_value <= self.value
        elif self.operator == OperatorType.IN:
            return col_value in self.value
        elif self.operator == OperatorType.NOT_IN:
            return col_value not in self.value
        elif self.operator == OperatorType.CONTAINS:
            return self.value in str(col_value)
        elif self.operator == OperatorType.NOT_CONTAINS:
            return self.value not in str(col_value)
        elif self.operator == OperatorType.IS_NULL:
            return col_value is None
        elif self.operator == OperatorType.IS_NOT_NULL:
            return col_value is not None
        else:
            raise ValueError(f"不支持的操作符: {self.operator}")


@dataclass
class PredictClause:
    """PREDICT子句"""
    aggregation: AggregationType
    target_column: ColumnReference
    time_window: Optional[TimeWindow] = None
    comparison_operator: Optional[OperatorType] = None
    comparison_value: Optional[Union[int, float]] = None

    def is_binary_classification(self) -> bool:
        """是否为二分类任务"""
        return self.comparison_operator is not None


@dataclass
class ForClause:
    """FOR子句"""
    entity_column: ColumnReference
    entity_values: List[Any] = field(default_factory=list)
    is_each: bool = False  # FOR EACH语法


@dataclass
class WhereClause:
    """WHERE子句"""
    conditions: List[Condition] = field(default_factory=list)
    logic: str = "AND"  # AND或OR

    def evaluate(self, row_data: Dict[str, Any]) -> bool:
        """评估WHERE条件"""
        if not self.conditions:
            return True
        
        if self.logic == "AND":
            return all(cond.evaluate(row_data) for cond in self.conditions)
        elif self.logic == "OR":
            return any(cond.evaluate(row_data) for cond in self.conditions)
        else:
            raise ValueError(f"不支持的逻辑操作: {self.logic}")


@dataclass
class PQLQuery:
    """完整的PQL查询"""
    predict_clause: PredictClause
    for_clause: ForClause
    where_clause: Optional[WhereClause] = None
    raw_query: Optional[str] = None

    def get_task_type(self) -> str:
        """推断任务类型"""
        agg = self.predict_clause.aggregation
        
        # 二分类
        if self.predict_clause.is_binary_classification():
            return "classification"
        
        # 基于聚合类型判断
        if agg in [AggregationType.COUNT, AggregationType.SUM, 
                   AggregationType.AVG, AggregationType.MIN, AggregationType.MAX]:
            return "regression"
        elif agg == AggregationType.EXISTS:
            return "classification"
        elif agg in [AggregationType.FIRST, AggregationType.LAST]:
            # 需要根据目标列类型判断
            return "classification"  # 默认分类
        elif agg == AggregationType.LIST_DISTINCT:
            return "link_prediction"
        else:
            raise ValueError(f"无法推断任务类型: {agg}")

    def get_target_info(self) -> Dict[str, Any]:
        """获取目标信息"""
        return {
            'table': self.predict_clause.target_column.table,
            'column': self.predict_clause.target_column.column,
            'aggregation': self.predict_clause.aggregation.value,
            'time_window': {
                'start': self.predict_clause.time_window.start_offset,
                'end': self.predict_clause.time_window.end_offset
            } if self.predict_clause.time_window else None
        }

    def get_entity_info(self) -> Dict[str, Any]:
        """获取实体信息"""
        return {
            'table': self.for_clause.entity_column.table,
            'column': self.for_clause.entity_column.column,
            'values': self.for_clause.entity_values,
            'is_each': self.for_clause.is_each
        }


class PQLQueryBuilder:
    """PQL查询构建器"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置构建器"""
        self._predict_clause = None
        self._for_clause = None
        self._where_clause = None
        return self

    def predict(self, 
                aggregation: Union[str, AggregationType],
                target: str,
                start_offset: Optional[int] = None,
                end_offset: Optional[int] = None) -> 'PQLQueryBuilder':
        """设置PREDICT子句"""
        if isinstance(aggregation, str):
            aggregation = AggregationType[aggregation.upper()]
        
        target_ref = ColumnReference.parse(target)
        
        time_window = None
        if start_offset is not None and end_offset is not None:
            time_window = TimeWindow(start_offset, end_offset)
        
        self._predict_clause = PredictClause(
            aggregation=aggregation,
            target_column=target_ref,
            time_window=time_window
        )
        
        return self

    def compare(self, operator: Union[str, OperatorType], value: Union[int, float]) -> 'PQLQueryBuilder':
        """添加比较操作（用于二分类）"""
        if self._predict_clause is None:
            raise ValueError("必须先调用predict()")
        
        if isinstance(operator, str):
            operator_map = {
                '=': OperatorType.EQ,
                '!=': OperatorType.NE,
                '>': OperatorType.GT,
                '>=': OperatorType.GE,
                '<': OperatorType.LT,
                '<=': OperatorType.LE
            }
            operator = operator_map.get(operator, OperatorType.EQ)
        
        self._predict_clause.comparison_operator = operator
        self._predict_clause.comparison_value = value
        
        return self

    def for_entity(self, entity: str, values: Optional[List[Any]] = None, each: bool = False) -> 'PQLQueryBuilder':
        """设置FOR子句"""
        entity_ref = ColumnReference.parse(entity)
        
        self._for_clause = ForClause(
            entity_column=entity_ref,
            entity_values=values or [],
            is_each=each
        )
        
        return self

    def where(self, column: str, operator: str, value: Any) -> 'PQLQueryBuilder':
        """添加WHERE条件"""
        if self._where_clause is None:
            self._where_clause = WhereClause()
        
        column_ref = ColumnReference.parse(column)
        
        operator_map = {
            '=': OperatorType.EQ,
            '!=': OperatorType.NE,
            '>': OperatorType.GT,
            '>=': OperatorType.GE,
            '<': OperatorType.LT,
            '<=': OperatorType.LE,
            'IN': OperatorType.IN,
            'NOT IN': OperatorType.NOT_IN,
            'CONTAINS': OperatorType.CONTAINS
        }
        
        op_type = operator_map.get(operator.upper(), OperatorType.EQ)
        
        condition = Condition(
            column=column_ref,
            operator=op_type,
            value=value
        )
        
        self._where_clause.conditions.append(condition)
        
        return self

    def build(self) -> PQLQuery:
        """构建查询对象"""
        if self._predict_clause is None:
            raise ValueError("缺少PREDICT子句")
        
        if self._for_clause is None:
            raise ValueError("缺少FOR子句")
        
        return PQLQuery(
            predict_clause=self._predict_clause,
            for_clause=self._for_clause,
            where_clause=self._where_clause
        )

    def to_string(self) -> str:
        """生成查询字符串"""
        query = self.build()
        parts = []
        
        # PREDICT子句
        pred = query.predict_clause
        predict_str = f"PREDICT {pred.aggregation.value}({pred.target_column}"
        
        if pred.time_window:
            predict_str += f", {pred.time_window.start_offset}, {pred.time_window.end_offset}"
        
        predict_str += ")"
        
        if pred.comparison_operator:
            predict_str += f" {pred.comparison_operator.value} {pred.comparison_value}"
        
        parts.append(predict_str)
        
        # FOR子句
        for_clause = query.for_clause
        for_str = "FOR"
        
        if for_clause.is_each:
            for_str += " EACH"
        
        for_str += f" {for_clause.entity_column}"
        
        if len(for_clause.entity_values) == 1:
            for_str += f" = {for_clause.entity_values[0]}"
        elif len(for_clause.entity_values) > 1:
            values_str = ", ".join(str(v) for v in for_clause.entity_values)
            for_str += f" IN ({values_str})"
        
        parts.append(for_str)
        
        # WHERE子句
        if query.where_clause and query.where_clause.conditions:
            where_parts = []
            for cond in query.where_clause.conditions:
                where_parts.append(
                    f"{cond.column} {cond.operator.value} {cond.value}"
                )
            
            where_str = f"WHERE {f' {query.where_clause.logic} '.join(where_parts)}"
            parts.append(where_str)
        
        return " ".join(parts)
