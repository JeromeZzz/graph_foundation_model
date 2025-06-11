"""
上下文生成器
动态生成用于上下文学习的示例
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import random
from collections import defaultdict

from data.graph_builder import HeterogeneousGraph
from data.samplers import BackwardLookingSampler, ForwardLookingSampler
from config.model_config import KumoRFMConfig


class ContextGenerator:
    """
    上下文生成器
    负责动态生成训练上下文和标签
    """

    def __init__(self,
                 config: KumoRFMConfig,
                 graph: HeterogeneousGraph,
                 backward_sampler: BackwardLookingSampler,
                 forward_sampler: ForwardLookingSampler):
        self.config = config
        self.graph = graph
        self.backward_sampler = backward_sampler
        self.forward_sampler = forward_sampler

        # 缓存已生成的上下文
        self.context_cache = {}
        self.cache_size = 10000

        # 时序索引
        self._build_temporal_index()

    def _build_temporal_index(self):
        """构建时序索引以加速上下文生成"""
        # 按表和时间组织节点
        self.nodes_by_table_time = defaultdict(lambda: defaultdict(list))

        for table_name, nodes in self.graph.nodes.items():
            for node in nodes:
                if node.timestamp:
                    date = node.timestamp.date()
                    self.nodes_by_table_time[table_name][date].append(node.id)

    def generate_context_batch(self,
                               entity_ids: List[int],
                               prediction_times: List[datetime],
                               task_config: Dict[str, Any],
                               num_context_per_entity: int = None) -> Dict[str, torch.Tensor]:
        """
        批量生成上下文

        entity_ids: 实体ID列表
        prediction_times: 对应的预测时间
        task_config: 任务配置
        num_context_per_entity: 每个实体的上下文数量

        返回: {
            'context_subgraphs': 上下文子图列表,
            'context_labels': 上下文标签,
            'context_masks': 有效上下文的掩码
        }
        """
        if num_context_per_entity is None:
            num_context_per_entity = self.config.num_context_examples

        batch_size = len(entity_ids)
        all_context_subgraphs = []
        all_context_labels = []
        all_context_masks = []

        for entity_id, pred_time in zip(entity_ids, prediction_times):
            # 生成单个实体的上下文
            context_data = self.generate_context_for_entity(
                entity_id, pred_time, task_config, num_context_per_entity
            )

            all_context_subgraphs.append(context_data['subgraphs'])
            all_context_labels.append(context_data['labels'])
            all_context_masks.append(context_data['mask'])

        # 整理为批次格式
        return {
            'context_subgraphs': all_context_subgraphs,
            'context_labels': torch.stack(all_context_labels),
            'context_masks': torch.stack(all_context_masks)
        }

    def generate_context_for_entity(self,
                                    entity_id: int,
                                    prediction_time: datetime,
                                    task_config: Dict[str, Any],
                                    num_context: int) -> Dict[str, Any]:
        """
        为单个实体生成上下文

        返回: {
            'subgraphs': 上下文子图列表,
            'labels': 标签张量,
            'mask': 有效掩码
        }
        """
        # 检查缓存
        cache_key = (entity_id, prediction_time, str(task_config))
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]

        # 获取实体信息
        entity_node = self.graph.node_id_map[entity_id]
        entity_table = entity_node.table

        # 采样策略
        context_strategy = task_config.get('context_strategy', 'temporal_proximity')

        if context_strategy == 'temporal_proximity':
            # 基于时间接近度采样
            context_candidates = self._sample_temporal_neighbors(
                entity_id, prediction_time, entity_table
            )
        elif context_strategy == 'structural_similarity':
            # 基于结构相似性采样
            context_candidates = self._sample_structural_neighbors(
                entity_id, prediction_time
            )
        elif context_strategy == 'mixed':
            # 混合策略
            temporal_candidates = self._sample_temporal_neighbors(
                entity_id, prediction_time, entity_table, num_context // 2
            )
            structural_candidates = self._sample_structural_neighbors(
                entity_id, prediction_time, num_context // 2
            )
            context_candidates = temporal_candidates + structural_candidates
        else:
            raise ValueError(f"未知的上下文策略: {context_strategy}")

        # 生成上下文子图和标签
        context_subgraphs = []
        context_labels = []
        valid_mask = []

        for candidate_id, candidate_time in context_candidates[:num_context]:
            # 生成历史子图
            try:
                subgraph, _ = self.backward_sampler.sample_subgraph(
                    center_node_id=candidate_id,
                    timestamp=candidate_time,
                    num_hops=2  # 上下文使用较小的跳数
                )

                # 生成标签
                label = self.forward_sampler.generate_label(
                    node_id=candidate_id,
                    prediction_time=candidate_time,
                    **task_config
                )

                if label is not None:
                    context_subgraphs.append(subgraph)
                    context_labels.append(label)
                    valid_mask.append(1)
                else:
                    # 无效标签，添加占位符
                    context_subgraphs.append(None)
                    context_labels.append(0)
                    valid_mask.append(0)

            except Exception as e:
                # 处理采样失败
                context_subgraphs.append(None)
                context_labels.append(0)
                valid_mask.append(0)

        # 填充到固定长度
        while len(context_subgraphs) < num_context:
            context_subgraphs.append(None)
            context_labels.append(0)
            valid_mask.append(0)

        # 转换为张量
        context_labels_tensor = torch.tensor(context_labels, dtype=torch.float32)
        valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool)

        result = {
            'subgraphs': context_subgraphs,
            'labels': context_labels_tensor,
            'mask': valid_mask_tensor
        }

        # 更新缓存
        if len(self.context_cache) < self.cache_size:
            self.context_cache[cache_key] = result

        return result

    def _sample_temporal_neighbors(self,
                                   entity_id: int,
                                   prediction_time: datetime,
                                   entity_table: str,
                                   num_samples: Optional[int] = None) -> List[Tuple[int, datetime]]:
        """
        基于时间接近度采样邻居
        """
        if num_samples is None:
            num_samples = self.config.num_context_examples * 2  # 过采样

        # 设置时间窗口
        lookback_days = self.backward_sampler.config.lookback_window or 365
        start_time = prediction_time - timedelta(days=lookback_days)

        candidates = []

        # 1. 同一实体的历史记录（自回归）
        entity_node = self.graph.node_id_map[entity_id]
        if entity_node.timestamp and start_time <= entity_node.timestamp < prediction_time:
            candidates.append((entity_id, entity_node.timestamp))

        # 2. 同表的其他实体
        for date in sorted(self.nodes_by_table_time[entity_table].keys(), reverse=True):
            if date >= prediction_time.date():
                continue
            if date < start_time.date():
                break

            for node_id in self.nodes_by_table_time[entity_table][date]:
                if node_id != entity_id:
                    node = self.graph.node_id_map[node_id]
                    candidates.append((node_id, node.timestamp))

            if len(candidates) >= num_samples:
                break

        # 3. 直接邻居
        neighbors = self.graph.get_neighbors(entity_id, max_timestamp=prediction_time)
        for neighbor_id in neighbors[:20]:  # 限制邻居数量
            neighbor_node = self.graph.node_id_map[neighbor_id]
            if neighbor_node.timestamp and start_time <= neighbor_node.timestamp < prediction_time:
                candidates.append((neighbor_id, neighbor_node.timestamp))

        # 按时间排序并去重
        candidates = list(set(candidates))
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:num_samples]

    def _sample_structural_neighbors(self,
                                     entity_id: int,
                                     prediction_time: datetime,
                                     num_samples: Optional[int] = None) -> List[Tuple[int, datetime]]:
        """
        基于结构相似性采样邻居
        """
        if num_samples is None:
            num_samples = self.config.num_context_examples

        candidates = []

        # 获取k跳邻居
        visited = {entity_id}
        current_level = {entity_id}

        for hop in range(3):  # 最多3跳
            next_level = set()

            for node_id in current_level:
                neighbors = self.graph.get_neighbors(node_id, max_timestamp=prediction_time)

                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_level.add(neighbor_id)

                        # 添加为候选
                        neighbor_node = self.graph.node_id_map[neighbor_id]
                        if neighbor_node.timestamp and neighbor_node.timestamp < prediction_time:
                            candidates.append((neighbor_id, neighbor_node.timestamp))

            current_level = next_level

            if len(candidates) >= num_samples:
                break

        # 随机采样
        if len(candidates) > num_samples:
            candidates = random.sample(candidates, num_samples)

        return candidates

    def generate_hard_negatives(self,
                                entity_id: int,
                                prediction_time: datetime,
                                positive_label: Any,
                                task_config: Dict[str, Any],
                                num_negatives: int = 5) -> List[Tuple[HeterogeneousGraph, Any]]:
        """
        生成困难负样本
        用于对比学习或提高模型区分能力
        """
        hard_negatives = []

        # 获取相似但标签不同的示例
        candidates = self._sample_temporal_neighbors(
            entity_id, prediction_time,
            self.graph.node_id_map[entity_id].table,
            num_samples=num_negatives * 3
        )

        for candidate_id, candidate_time in candidates:
            # 生成子图
            subgraph, _ = self.backward_sampler.sample_subgraph(
                center_node_id=candidate_id,
                timestamp=candidate_time,
                num_hops=2
            )

            # 生成标签
            label = self.forward_sampler.generate_label(
                node_id=candidate_id,
                prediction_time=candidate_time,
                **task_config
            )

            # 检查是否为负样本
            if label is not None and label != positive_label:
                hard_negatives.append((subgraph, label))

                if len(hard_negatives) >= num_negatives:
                    break

        return hard_negatives

    def augment_context(self,
                        context_subgraphs: List[HeterogeneousGraph],
                        augmentation_type: str = "dropout") -> List[HeterogeneousGraph]:
        """
        上下文增强
        通过数据增强提高模型泛化能力
        """
        augmented = []

        for subgraph in context_subgraphs:
            if subgraph is None:
                augmented.append(None)
                continue

            if augmentation_type == "dropout":
                # 随机丢弃一些节点
                aug_subgraph = self._dropout_nodes(subgraph, dropout_rate=0.1)

            elif augmentation_type == "noise":
                # 添加特征噪声
                aug_subgraph = self._add_feature_noise(subgraph, noise_level=0.05)

            elif augmentation_type == "subgraph":
                # 采样子图的子图
                aug_subgraph = self._sample_subsubgraph(subgraph, sample_rate=0.8)

            else:
                aug_subgraph = subgraph

            augmented.append(aug_subgraph)

        return augmented

    def _dropout_nodes(self, subgraph: HeterogeneousGraph,
                       dropout_rate: float) -> HeterogeneousGraph:
        """随机丢弃节点"""
        # 简化实现：返回原图
        # 实际实现需要创建新的子图并随机删除节点
        return subgraph

    def _add_feature_noise(self, subgraph: HeterogeneousGraph,
                           noise_level: float) -> HeterogeneousGraph:
        """添加特征噪声"""
        # 简化实现：返回原图
        # 实际实现需要复制子图并向数值特征添加高斯噪声
        return subgraph

    def _sample_subsubgraph(self, subgraph: HeterogeneousGraph,
                            sample_rate: float) -> HeterogeneousGraph:
        """采样子图的子图"""
        # 简化实现：返回原图
        # 实际实现需要随机采样节点子集并构建新子图
        return subgraph