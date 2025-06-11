"""
PQL查询验证器
验证查询的语法和语义正确性
"""

from typing import Dict, List, Optional, Set, Tuple
import re
from datetime import datetime

from .query import PQLQuery, AggregationType, OperatorType
from data.database import RelationalDatabase


class ValidationError(Exception):
    """验证错误"""
    pass


class PQLValidator:
    """
    PQL查询验证器
    """

    def __init__(self, database: Optional[RelationalDatabase] = None):
        self.database = database
        
        # 支持的聚合函数
        self.supported_aggregations = set(agg.value for agg in AggregationType)
        
        # 支持的操作符
        self.supported_operators = set(op.value for op in OperatorType)

    def validate(self, query: PQLQuery) -> List[str]:
        """
        验证查询
        
        query: PQL查询对象
        
        返回: 错误信息列表
        """
        errors = []
        
        # 1. 验证PREDICT子句
        predict_errors = self._validate_predict_clause(query.predict_clause)
        errors.extend(predict_errors)
        
        # 2. 验证FOR子句
        for_errors = self._validate_for_clause(query.for_clause)
        errors.extend(for_errors)
        
        # 3. 验证WHERE子句
        if query.where_clause:
            where_errors = self._validate_where_clause(query.where_clause)
            errors.extend(where_errors)
        
        # 4. 如果有数据库，验证表和列
        if self.database:
            schema_errors = self._validate_against_schema(query)
            errors.extend(schema_errors)
        
        # 5. 验证语义一致性
        semantic_errors = self._validate_semantics(query)
        errors.extend(semantic_errors)
        
        return errors

    def _validate_predict_clause(self, predict_clause) -> List[str]:
        """验证PREDICT子句"""
        errors = []
        
        # 验证聚合函数
        if predict_clause.aggregation.value not in self.supported_aggregations:
            errors.append(
                f"不支持的聚合函数: {predict_clause.aggregation.value}"
            )
        
        # 验证时间窗口
        if predict_clause.time_window:
            tw = predict_clause.time_window
            if tw.start_offset > tw.end_offset:
                errors.append(
                    f"无效的时间窗口: 开始偏移 {tw.start_offset} > 结束偏移 {tw.end_offset}"
                )
            
            # 检查时间窗口的合理性
            if tw.end_offset - tw.start_offset > 365:
                errors.append(
                    f"时间窗口过大: {tw.end_offset - tw.start_offset} 天"
                )
        
        # 验证比较操作
        if predict_clause.comparison_operator:
            # 只有某些聚合函数支持比较
            if predict_clause.aggregation not in [
                AggregationType.COUNT, AggregationType.SUM, 
                AggregationType.AVG, AggregationType.EXISTS
            ]:
                errors.append(
                    f"聚合函数 {predict_clause.aggregation.value} 不支持比较操作"
                )
            
            # 验证比较值
            if predict_clause.comparison_value is None:
                errors.append("比较操作需要提供比较值")
        
        # 特定聚合函数的验证
        if predict_clause.aggregation == AggregationType.LIST_DISTINCT:
            if predict_clause.time_window is None:
                errors.append("LIST_DISTINCT 需要时间窗口")
        
        return errors

    def _validate_for_clause(self, for_clause) -> List[str]:
        """验证FOR子句"""
        errors = []
        
        # 验证实体列
        if not for_clause.entity_column.column:
            errors.append("FOR子句需要指定实体列")
        
        # 验证实体值
        if for_clause.is_each and not for_clause.entity_values:
            errors.append("FOR EACH子句需要提供实体值列表")
        
        # 检查实体值类型一致性
        if for_clause.entity_values:
            value_types = set(type(v) for v in for_clause.entity_values)
            if len(value_types) > 1:
                errors.append(
                    f"实体值类型不一致: {[t.__name__ for t in value_types]}"
                )
        
        return errors

    def _validate_where_clause(self, where_clause) -> List[str]:
        """验证WHERE子句"""
        errors = []
        
        if not where_clause.conditions:
            errors.append("WHERE子句没有条件")
            return errors
        
        # 验证每个条件
        for i, condition in enumerate(where_clause.conditions):
            # 验证操作符
            if condition.operator.value not in self.supported_operators:
                errors.append(
                    f"条件 {i+1}: 不支持的操作符 {condition.operator.value}"
                )
            
            # 验证值类型与操作符的兼容性
            if condition.operator in [OperatorType.IN, OperatorType.NOT_IN]:
                if not isinstance(condition.value, (list, tuple, set)):
                    errors.append(
                        f"条件 {i+1}: {condition.operator.value} 操作符需要列表值"
                    )
            
            if condition.operator in [OperatorType.GT, OperatorType.GE, 
                                     OperatorType.LT, OperatorType.LE]:
                if not isinstance(condition.value, (int, float, datetime)):
                    errors.append(
                        f"条件 {i+1}: 比较操作符需要数值或时间值"
                    )
        
        # 验证逻辑操作符
        if where_clause.logic not in ["AND", "OR"]:
            errors.append(f"不支持的逻辑操作符: {where_clause.logic}")
        
        return errors

    def _validate_against_schema(self, query: PQLQuery) -> List[str]:
        """验证查询是否符合数据库模式"""
        errors = []
        
        # 获取所有表名
        table_names = set(self.database.tables.keys())
        
        # 验证目标表和列
        target_col = query.predict_clause.target_column
        if target_col.table:
            if target_col.table not in table_names:
                errors.append(f"目标表 '{target_col.table}' 不存在")
            else:
                # 验证列
                table = self.database.get_table(target_col.table)
                if target_col.column != '*' and target_col.column not in table.data.columns:
                    errors.append(
                        f"列 '{target_col.column}' 在表 '{target_col.table}' 中不存在"
                    )
        
        # 验证实体表和列
        entity_col = query.for_clause.entity_column
        if entity_col.table:
            if entity_col.table not in table_names:
                errors.append(f"实体表 '{entity_col.table}' 不存在")
            else:
                table = self.database.get_table(entity_col.table)
                if entity_col.column not in table.data.columns:
                    errors.append(
                        f"列 '{entity_col.column}' 在表 '{entity_col.table}' 中不存在"
                    )
        
        # 验证WHERE子句中的列
        if query.where_clause:
            for condition in query.where_clause.conditions:
                col = condition.column
                if col.table and col.table not in table_names:
                    errors.append(f"WHERE子句: 表 '{col.table}' 不存在")
                elif col.table:
                    table = self.database.get_table(col.table)
                    if col.column not in table.data.columns:
                        errors.append(
                            f"WHERE子句: 列 '{col.column}' 在表 '{col.table}' 中不存在"
                        )
        
        # 验证表之间的关系
        errors.extend(self._validate_relationships(query))
        
        return errors

    def _validate_relationships(self, query: PQLQuery) -> List[str]:
        """验证查询中涉及的表之间是否有关系"""
        errors = []
        
        # 收集查询中涉及的所有表
        tables_used = set()
        
        # 目标表
        if query.predict_clause.target_column.table:
            tables_used.add(query.predict_clause.target_column.table)
        
        # 实体表
        if query.for_clause.entity_column.table:
            tables_used.add(query.for_clause.entity_column.table)
        
        # WHERE子句中的表
        if query.where_clause:
            for condition in query.where_clause.conditions:
                if condition.column.table:
                    tables_used.add(condition.column.table)
        
        # 检查表之间是否有路径
        if len(tables_used) > 1:
            # 简化检查：确保实体表能到达目标表
            entity_table = query.for_clause.entity_column.table
            target_table = query.predict_clause.target_column.table
            
            if entity_table and target_table and entity_table != target_table:
                path = self.database.get_join_path(entity_table, target_table)
                if path is None:
                    errors.append(
                        f"表 '{entity_table}' 和 '{target_table}' 之间没有关系路径"
                    )
        
        return errors

    def _validate_semantics(self, query: PQLQuery) -> List[str]:
        """验证查询的语义一致性"""
        errors = []
        
        # 验证聚合函数与任务类型的一致性
        task_type = query.get_task_type()
        aggregation = query.predict_clause.aggregation
        
        # LIST_DISTINCT只用于链接预测
        if aggregation == AggregationType.LIST_DISTINCT and task_type != "link_prediction":
            errors.append(
                "LIST_DISTINCT 聚合函数只能用于链接预测任务"
            )
        
        # EXISTS通常用于分类
        if aggregation == AggregationType.EXISTS and not query.predict_clause.is_binary_classification():
            errors.append(
                "EXISTS 聚合函数通常需要比较操作来形成二分类任务"
            )
        
        # 时间窗口的必要性
        if aggregation in [AggregationType.COUNT, AggregationType.SUM, 
                          AggregationType.AVG, AggregationType.LIST_DISTINCT]:
            if not query.predict_clause.time_window:
                errors.append(
                    f"{aggregation.value} 聚合函数建议指定时间窗口"
                )
        
        return errors

    def validate_query_string(self, query_string: str) -> Tuple[bool, List[str]]:
        """
        验证查询字符串（不解析）
        
        返回: (是否有效, 错误列表)
        """
        errors = []
        
        # 基本语法检查
        if not query_string.strip():
            errors.append("查询字符串为空")
            return False, errors
        
        # 检查必需的关键字
        if "PREDICT" not in query_string.upper():
            errors.append("缺少 PREDICT 关键字")
        
        if "FOR" not in query_string.upper():
            errors.append("缺少 FOR 关键字")
        
        # 检查括号匹配
        open_parens = query_string.count('(')
        close_parens = query_string.count(')')
        if open_parens != close_parens:
            errors.append(f"括号不匹配: {open_parens} 个 '(' 和 {close_parens} 个 ')'")
        
        # 检查聚合函数格式
        agg_pattern = r'PREDICT\s+(\w+)\s*\('
        match = re.search(agg_pattern, query_string, re.IGNORECASE)
        if match:
            agg_func = match.group(1).upper()
            if agg_func not in self.supported_aggregations:
                errors.append(f"不支持的聚合函数: {agg_func}")
        else:
            errors.append("PREDICT 子句格式错误")
        
        return len(errors) == 0, errors

    def suggest_corrections(self, query: PQLQuery, errors: List[str]) -> List[str]:
        """
        基于验证错误提供修正建议
        
        返回: 建议列表
        """
        suggestions = []
        
        for error in errors:
            if "时间窗口" in error and "需要" in error:
                suggestions.append(
                    "建议添加时间窗口，例如: (column, 0, 7) 表示未来7天"
                )
            
            elif "不存在" in error and "表" in error:
                if self.database:
                    available_tables = list(self.database.tables.keys())
                    suggestions.append(
                        f"可用的表包括: {', '.join(available_tables)}"
                    )
            
            elif "不存在" in error and "列" in error:
                # 提取表名
                match = re.search(r"表 '(\w+)'", error)
                if match and self.database:
                    table_name = match.group(1)
                    if table_name in self.database.tables:
                        columns = list(self.database.tables[table_name].data.columns)
                        suggestions.append(
                            f"表 '{table_name}' 的可用列: {', '.join(columns[:5])}..."
                        )
            
            elif "比较操作" in error:
                suggestions.append(
                    "如果需要二分类，请添加比较操作，例如: > 0"
                )
        
        return suggestions
