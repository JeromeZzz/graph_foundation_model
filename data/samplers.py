"""
图采样器
实现前向和后向的时序图采样
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import random
from collections import defaultdict

from .graph_builder import HeterogeneousGraph, Node
from config.model_config import SamplerConfig


class TemporalSampler:
    """
    时序图采样器基类
    """

    def __init__(self, graph: HeterogeneousGraph, config: SamplerConfig):
        self.graph = graph
        self.config = config

        # 构建时序索引以加速采样
        self._build_temporal_index()

    def _build_temporal_index(self):
        """构建时序索引"""
        # 按时间戳组织节点
        self.nodes_by_time = defaultdict(list)
        for table_nodes in self.graph.nodes.values():
            for node in table_nodes:
                if node.timestamp:
                    # 将时间戳转换为日期（忽略时间）
                    date = node.timestamp.date()
                    self.nodes_by_time[date].append(node.id)

        # 按时间戳组织边
        self.edges_by_time = defaultdict(list)
        for edge_type, edges in self.graph.edges.items():
            for edge in edges:
                if edge.timestamp:
                    date = edge.timestamp.date()
                    self.edges_by_time[date].append((edge_type, edge))

    def sample_neighbors(self, node_id: int, timestamp: datetime,
                         num_neighbors: int, strategy: str = "uniform") -> List[int]:
        """
        采样节点的邻居
        strategy: uniform（均匀）, temporal（时序优先）, importance（重要性）
        """
        # 获取所有符合时间条件的邻居
        all_neighbors = self.graph.get_neighbors(node_id, max_timestamp=timestamp)

        if len(all_neighbors) <= num_neighbors:
            return all_neighbors

        if strategy == "uniform":
            return random.sample(all_neighbors, num_neighbors)

        elif strategy == "temporal":
            # 按时间接近度排序
            node = self.graph.node_id_map[node_id]
            neighbor_times = []

            for n_id in all_neighbors:
                n_node = self.graph.node_id_map[n_id]
                if n_node.timestamp:
                    time_diff = abs((timestamp - n_node.timestamp).total_seconds())
                    neighbor_times.append((n_id, time_diff))
                else:
                    neighbor_times.append((n_id, float('inf')))

            # 按时间差排序，选择最近的
            neighbor_times.sort(key=lambda x: x[1])
            return [n_id for n_id, _ in neighbor_times[:num_neighbors]]

        elif strategy == "importance":
            # 基于度数的重要性采样
            neighbor_degrees = []
            for n_id in all_neighbors:
                degree = len(self.graph.get_neighbors(n_id, max_timestamp=timestamp))
                neighbor_degrees.append((n_id, degree))

            # 按度数排序，选择度数最高的
            neighbor_degrees.sort(key=lambda x: x[1], reverse=True)
            return [n_id for n_id, _ in neighbor_degrees[:num_neighbors]]

        else:
            raise ValueError(f"未知的采样策略: {strategy}")


class BackwardLookingSampler(TemporalSampler):
    """
    后向采样器
    用于生成输入子图（只包含时间戳早于预测时间的数据）
    """

    def sample_subgraph(self, center_node_id: int, timestamp: datetime,
                        num_hops: int) -> Tuple[HeterogeneousGraph, List[int]]:
        """
        采样子图
        返回：(子图, 采样的节点ID列表)
        """
        subgraph = HeterogeneousGraph()
        sampled_nodes = set()

        # 多跳采样
        current_level = {center_node_id}

        for hop in range(num_hops + 1):
            next_level = set()

            for node_id in current_level:
                if node_id in sampled_nodes:
                    continue

                # 检查节点时间戳
                node = self.graph.node_id_map[node_id]
                if node.timestamp and node.timestamp > timestamp:
                    continue

                sampled_nodes.add(node_id)

                # 添加节点到子图
                subgraph.add_node(
                    table=node.table,
                    primary_key=node.primary_key,
                    features=node.features,
                    timestamp=node.timestamp
                )

                if hop < num_hops:
                    # 采样邻居
                    neighbors = self.sample_neighbors(
                        node_id=node_id,
                        timestamp=timestamp,
                        num_neighbors=self.config.max_neighbors,
                        strategy=self.config.neighbor_sampling_strategy
                    )

                    # 自适应采样
                    if self.config.use_adaptive_sampling and len(neighbors) < self.config.min_neighbors_per_hop:
                        # 如果邻居太少，扩大搜索范围
                        all_neighbors = self.graph.get_neighbors(node_id, max_timestamp=timestamp)
                        additional = min(
                            self.config.min_neighbors_per_hop - len(neighbors),
                            len(all_neighbors) - len(neighbors)
                        )
                        if additional > 0:
                            remaining = [n for n in all_neighbors if n not in neighbors]
                            neighbors.extend(random.sample(remaining, additional))

                    next_level.update(neighbors)

            current_level = next_level

            # 检查是否达到最大节点数
            if len(sampled_nodes) >= self.config.max_sampled_nodes:
                break

        # 添加边
        for edge_type, edges in self.graph.edges.items():
            for edge in edges:
                # 时间过滤
                if edge.timestamp and edge.timestamp > timestamp:
                    continue

                # 只添加两个节点都在子图中的边
                if edge.source in sampled_nodes and edge.target in sampled_nodes:
                    # 获取子图中的节点ID
                    source_node = self.graph.node_id_map[edge.source]
                    target_node = self.graph.node_id_map[edge.target]

                    new_source_id = subgraph.pk_to_node_id[(source_node.table, source_node.primary_key)]
                    new_target_id = subgraph.pk_to_node_id[(target_node.table, target_node.primary_key)]

                    subgraph.add_edge(
                        source_id=new_source_id,
                        target_id=new_target_id,
                        edge_type=edge_type,
                        timestamp=edge.timestamp,
                        features=edge.features
                    )

        return subgraph, list(sampled_nodes)

    def sample_temporal_neighbors(self, node_id: int, timestamp: datetime,
                                  window_days: int = 30) -> List[Tuple[int, datetime]]:
        """
        采样时间窗口内的邻居节点
        用于生成历史上下文
        """
        start_time = timestamp - timedelta(days=window_days)
        temporal_neighbors = []

        # 获取时间窗口内的所有邻居
        neighbors = self.graph.get_neighbors(node_id, max_timestamp=timestamp)

        for n_id in neighbors:
            node = self.graph.node_id_map[n_id]
            if node.timestamp and start_time <= node.timestamp <= timestamp:
                temporal_neighbors.append((n_id, node.timestamp))

        # 按时间排序
        temporal_neighbors.sort(key=lambda x: x[1])

        return temporal_neighbors


class ForwardLookingSampler(TemporalSampler):
    """
    前向采样器
    用于生成标签（包含预测时间之后的数据）
    """

    def sample_future_events(self, node_id: int, start_time: datetime,
                             end_time: datetime, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        采样未来事件
        用于生成训练标签
        """
        future_events = []

        # 查找相关的边
        for edge_type, edges in self.graph.edges.items():
            # 过滤事件类型
            if event_type and edge_type != event_type:
                continue

            for edge in edges:
                # 时间过滤
                if not edge.timestamp:
                    continue
                if not (start_time <= edge.timestamp <= end_time):
                    continue

                # 检查是否涉及目标节点
                if edge.source == node_id or edge.target == node_id:
                    event = {
                        'edge_type': edge_type,
                        'timestamp': edge.timestamp,
                        'source': edge.source,
                        'target': edge.target,
                        'features': edge.features
                    }
                    future_events.append(event)

        # 按时间排序
        future_events.sort(key=lambda x: x['timestamp'])

        return future_events

    def generate_label(self, node_id: int, prediction_time: datetime,
                       task_type: str, aggregation_window: int = 7,
                       target_column: Optional[str] = None) -> Any:
        """
        生成标签
        task_type: count（计数）, sum（求和）, exists（存在性）, next（下一个）
        aggregation_window: 聚合窗口（天数）
        """
        end_time = prediction_time + timedelta(days=aggregation_window)

        if task_type == "count":
            # 计数任务：统计时间窗口内的事件数
            events = self.sample_future_events(node_id, prediction_time, end_time)
            return len(events)

        elif task_type == "sum":
            # 求和任务：对特定特征求和
            if not target_column:
                raise ValueError("求和任务需要指定target_column")

            events = self.sample_future_events(node_id, prediction_time, end_time)
            total = 0
            for event in events:
                if event['features'] and target_column in event['features']:
                    total += float(event['features'][target_column])
            return total

        elif task_type == "exists":
            # 存在性任务：是否有事件发生
            events = self.sample_future_events(node_id, prediction_time, end_time)
            return 1 if events else 0

        elif task_type == "next":
            # 下一个事件任务：预测下一个事件的目标
            events = self.sample_future_events(node_id, prediction_time, end_time)
            if events:
                first_event = events[0]
                # 返回事件的另一端节点
                if first_event['source'] == node_id:
                    return first_event['target']
                else:
                    return first_event['source']
            return None

        else:
            raise ValueError(f"未知的任务类型: {task_type}")


class OnlineContextGenerator:
    """
    在线上下文生成器
    动态生成训练上下文和标签
    """

    def __init__(self, graph: HeterogeneousGraph,
                 backward_sampler: BackwardLookingSampler,
                 forward_sampler: ForwardLookingSampler):
        self.graph = graph
        self.backward_sampler = backward_sampler
        self.forward_sampler = forward_sampler

    def generate_context(self, entity_id: int, prediction_time: datetime,
                         num_context_examples: int, task_config: Dict[str, Any]) -> Tuple[
        List[HeterogeneousGraph], List[Any]]:
        """
        生成上下文示例
        返回：(上下文子图列表, 对应标签列表)
        """
        context_graphs = []
        context_labels = []

        # 获取实体的历史时间戳
        node = self.graph.node_id_map[entity_id]

        # 采样历史时间点
        if self.backward_sampler.config.lookback_window:
            start_time = prediction_time - timedelta(days=self.backward_sampler.config.lookback_window)
        else:
            start_time = datetime.min

        # 获取时间范围内的邻居作为上下文候选
        temporal_neighbors = self.backward_sampler.sample_temporal_neighbors(
            entity_id, prediction_time,
            window_days=self.backward_sampler.config.lookback_window or 365
        )

        # 随机采样上下文示例
        if len(temporal_neighbors) > num_context_examples:
            sampled_neighbors = random.sample(temporal_neighbors, num_context_examples)
        else:
            sampled_neighbors = temporal_neighbors

        for neighbor_id, neighbor_time in sampled_neighbors:
            # 生成历史子图
            subgraph, _ = self.backward_sampler.sample_subgraph(
                center_node_id=neighbor_id,
                timestamp=neighbor_time,
                num_hops=2  # 上下文使用较小的跳数
            )

            # 生成对应的标签
            label = self.forward_sampler.generate_label(
                node_id=neighbor_id,
                prediction_time=neighbor_time,
                **task_config
            )

            if label is not None:  # 过滤掉无效标签
                context_graphs.append(subgraph)
                context_labels.append(label)

        return context_graphs, context_labels