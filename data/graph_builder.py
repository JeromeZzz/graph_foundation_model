"""
图构建器
将关系型数据库转换为时序异构图
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass

from .database import RelationalDatabase, Table


@dataclass
class Node:
    """图中的节点"""
    id: int  # 全局唯一ID
    table: str  # 所属表
    primary_key: Any  # 主键值
    features: Dict[str, Any]  # 节点特征
    timestamp: Optional[datetime] = None  # 时间戳（如果有）


@dataclass
class Edge:
    """图中的边"""
    source: int  # 源节点ID
    target: int  # 目标节点ID
    edge_type: str  # 边类型（表1_列1__表2_列2）
    timestamp: Optional[datetime] = None  # 时间戳（如果有）
    features: Optional[Dict[str, Any]] = None  # 边特征


class HeterogeneousGraph:
    """
    异构图表示
    支持多种节点类型和边类型
    """

    def __init__(self):
        self.nodes: Dict[str, List[Node]] = defaultdict(list)  # 按表分组的节点
        self.edges: Dict[str, List[Edge]] = defaultdict(list)  # 按类型分组的边

        # 索引结构
        self.node_id_map: Dict[int, Node] = {}  # ID到节点的映射
        self.pk_to_node_id: Dict[Tuple[str, Any], int] = {}  # (表名, 主键)到节点ID的映射

        self._next_node_id = 0

    def add_node(self, table: str, primary_key: Any,
                 features: Dict[str, Any], timestamp: Optional[datetime] = None) -> int:
        """添加节点并返回节点ID"""
        # 检查节点是否已存在
        key = (table, primary_key)
        if key in self.pk_to_node_id:
            return self.pk_to_node_id[key]

        # 创建新节点
        node_id = self._next_node_id
        self._next_node_id += 1

        node = Node(
            id=node_id,
            table=table,
            primary_key=primary_key,
            features=features,
            timestamp=timestamp
        )

        self.nodes[table].append(node)
        self.node_id_map[node_id] = node
        self.pk_to_node_id[key] = node_id

        return node_id

    def add_edge(self, source_id: int, target_id: int, edge_type: str,
                 timestamp: Optional[datetime] = None, features: Optional[Dict[str, Any]] = None):
        """添加边"""
        # 验证节点存在
        if source_id not in self.node_id_map:
            raise ValueError(f"源节点 {source_id} 不存在")
        if target_id not in self.node_id_map:
            raise ValueError(f"目标节点 {target_id} 不存在")

        edge = Edge(
            source=source_id,
            target=target_id,
            edge_type=edge_type,
            timestamp=timestamp,
            features=features
        )

        self.edges[edge_type].append(edge)

    def get_neighbors(self, node_id: int, edge_types: Optional[List[str]] = None,
                      max_timestamp: Optional[datetime] = None) -> List[int]:
        """
        获取节点的邻居
        edge_types: 限定边类型，None表示所有类型
        max_timestamp: 时间过滤，只返回早于此时间的边
        """
        neighbors = []

        edge_types_to_check = edge_types if edge_types else list(self.edges.keys())

        for edge_type in edge_types_to_check:
            for edge in self.edges[edge_type]:
                # 时间过滤
                if max_timestamp and edge.timestamp and edge.timestamp > max_timestamp:
                    continue

                if edge.source == node_id:
                    neighbors.append(edge.target)
                elif edge.target == node_id:
                    neighbors.append(edge.source)

        return list(set(neighbors))  # 去重

    def to_pytorch_geometric(self) -> Dict[str, torch.Tensor]:
        """
        转换为PyTorch Geometric格式
        返回边索引和节点特征
        """
        # 收集所有边
        all_edges = []
        edge_attr = []

        for edge_type, edges in self.edges.items():
            for edge in edges:
                all_edges.append([edge.source, edge.target])
                # 这里可以添加边特征编码

        if all_edges:
            edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return {
            'edge_index': edge_index,
            'num_nodes': len(self.node_id_map)
        }


class GraphBuilder:
    """
    将关系型数据库转换为图的构建器
    """

    def __init__(self, database: RelationalDatabase):
        self.database = database

    def build_full_graph(self,
                         timestamp_columns: Optional[Dict[str, str]] = None) -> HeterogeneousGraph:
        """
        构建完整的异构图
        timestamp_columns: 每个表的时间戳列名映射
        """
        graph = HeterogeneousGraph()

        # 1. 为每个表的每行创建节点
        for table_name, table in self.database.tables.items():
            timestamp_col = timestamp_columns.get(table_name) if timestamp_columns else None

            for idx, row in table.data.iterrows():
                # 提取特征
                features = {}
                timestamp = None

                for col in table.columns:
                    value = row[col.name]

                    # 处理时间戳
                    if timestamp_col and col.name == timestamp_col:
                        timestamp = pd.to_datetime(value) if pd.notna(value) else None

                    # 存储特征
                    features[col.name] = value

                # 添加节点
                if table.primary_key:
                    pk_value = row[table.primary_key]
                else:
                    pk_value = idx  # 使用行索引作为主键

                graph.add_node(
                    table=table_name,
                    primary_key=pk_value,
                    features=features,
                    timestamp=timestamp
                )

        # 2. 基于外键关系创建边
        for t1, c1, t2, c2 in self.database.relationships:
            edge_type = f"{t1}_{c1}__{t2}_{c2}"

            # 获取两个表的数据
            table1_data = self.database.tables[t1].data
            table2_data = self.database.tables[t2].data

            # 创建连接
            for idx1, row1 in table1_data.iterrows():
                fk_value = row1[c1]
                if pd.isna(fk_value):
                    continue

                # 在表2中查找匹配的行
                matching_rows = table2_data[table2_data[c2] == fk_value]

                for idx2, row2 in matching_rows.iterrows():
                    # 获取节点ID
                    pk1 = row1[self.database.tables[t1].primary_key] if self.database.tables[t1].primary_key else idx1
                    pk2 = row2[self.database.tables[t2].primary_key] if self.database.tables[t2].primary_key else idx2

                    node1_id = graph.pk_to_node_id.get((t1, pk1))
                    node2_id = graph.pk_to_node_id.get((t2, pk2))

                    if node1_id is not None and node2_id is not None:
                        # 获取边的时间戳（使用两个节点中较晚的时间）
                        node1 = graph.node_id_map[node1_id]
                        node2 = graph.node_id_map[node2_id]

                        edge_timestamp = None
                        if node1.timestamp and node2.timestamp:
                            edge_timestamp = max(node1.timestamp, node2.timestamp)
                        elif node1.timestamp:
                            edge_timestamp = node1.timestamp
                        elif node2.timestamp:
                            edge_timestamp = node2.timestamp

                        graph.add_edge(
                            source_id=node1_id,
                            target_id=node2_id,
                            edge_type=edge_type,
                            timestamp=edge_timestamp
                        )

        return graph

    def get_subgraph(self, graph: HeterogeneousGraph,
                     center_node_id: int, num_hops: int,
                     max_timestamp: Optional[datetime] = None) -> HeterogeneousGraph:
        """
        提取子图
        center_node_id: 中心节点ID
        num_hops: 跳数
        max_timestamp: 时间限制
        """
        subgraph = HeterogeneousGraph()
        visited = set()

        # BFS遍历
        current_level = {center_node_id}

        for hop in range(num_hops + 1):
            next_level = set()

            for node_id in current_level:
                if node_id in visited:
                    continue

                visited.add(node_id)

                # 添加节点到子图
                node = graph.node_id_map[node_id]
                new_node_id = subgraph.add_node(
                    table=node.table,
                    primary_key=node.primary_key,
                    features=node.features,
                    timestamp=node.timestamp
                )

                if hop < num_hops:
                    # 获取邻居
                    neighbors = graph.get_neighbors(node_id, max_timestamp=max_timestamp)
                    next_level.update(neighbors)

            current_level = next_level

        # 添加边（只包含子图中存在的节点之间的边）
        for edge_type, edges in graph.edges.items():
            for edge in edges:
                if edge.source in visited and edge.target in visited:
                    if max_timestamp and edge.timestamp and edge.timestamp > max_timestamp:
                        continue

                    # 获取子图中的节点ID
                    source_node = graph.node_id_map[edge.source]
                    target_node = graph.node_id_map[edge.target]

                    new_source_id = subgraph.pk_to_node_id[(source_node.table, source_node.primary_key)]
                    new_target_id = subgraph.pk_to_node_id[(target_node.table, target_node.primary_key)]

                    subgraph.add_edge(
                        source_id=new_source_id,
                        target_id=new_target_id,
                        edge_type=edge.edge_type,
                        timestamp=edge.timestamp,
                        features=edge.features
                    )

        return subgraph