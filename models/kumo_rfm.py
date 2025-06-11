"""
KumoRFM主模型
整合所有组件的完整模型实现
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging

from config.model_config import KumoRFMConfig, ColumnEncoderConfig
from .encoders.column_encoder import MultiModalColumnEncoder
from .encoders.table_encoder import TableEncoder
from .encoders.positional_encoding import CombinedPositionalEncoder
from .transformers.relational_graph_transformer import HeterogeneousRelationalGraphTransformer
from .icl.icl_module import InContextLearningModule
from .icl.context_generator import ContextGenerator
from data.graph_builder import HeterogeneousGraph
from data.samplers import BackwardLookingSampler, ForwardLookingSampler

logger = logging.getLogger(__name__)


class KumoRFM(nn.Module):
    """
    KumoRFM - 关系型基础模型

    支持：
    1. 任意关系型数据库结构
    2. 多种预测任务（分类、回归、链接预测）
    3. 上下文学习（无需任务特定训练）
    4. 可解释性
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config

        # 列编码器配置
        self.column_encoder_config = ColumnEncoderConfig()

        # 初始化组件（延迟初始化，需要数据库元数据）
        self.column_encoder = None
        self.table_encoder = None
        self.positional_encoder = None
        self.graph_transformer = None
        self.icl_module = None

        # 是否已初始化
        self.initialized = False

        logger.info("KumoRFM模型已创建")

    def initialize(self,
                   column_metadata: Dict[str, Dict[str, Any]],
                   table_metadata: Dict[str, Dict[str, Any]],
                   num_node_types: int,
                   num_edge_types: int):
        """
        使用数据库元数据初始化模型

        column_metadata: 列元数据
        table_metadata: 表元数据
        num_node_types: 节点类型数（表数）
        num_edge_types: 边类型数（关系数）
        """
        logger.info(f"初始化KumoRFM: {num_node_types}种节点类型, {num_edge_types}种边类型")

        # 1. 列编码器
        self.column_encoder = MultiModalColumnEncoder(
            config=self.column_encoder_config,
            column_metadata=column_metadata,
            text_model_name=self.config.text_encoder_model
        )

        # 2. 表编码器
        self.table_encoder = TableEncoder(self.config)

        # 3. 位置编码器
        self.positional_encoder = CombinedPositionalEncoder(
            config=self.config,
            num_node_types=num_node_types
        )

        # 4. 关系图Transformer
        self.graph_transformer = HeterogeneousRelationalGraphTransformer(
            config=self.config,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types
        )

        # 5. 上下文学习模块
        self.icl_module = InContextLearningModule(self.config)

        self.initialized = True
        logger.info("KumoRFM初始化完成")

    def forward(self,
                query_graph: HeterogeneousGraph,
                context_graphs: List[HeterogeneousGraph],
                context_labels: torch.Tensor,
                task_type: str,
                prediction_time: Optional[datetime] = None,
                label_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        query_graph: 查询图（要预测的）
        context_graphs: 上下文图列表
        context_labels: 上下文标签
        task_type: 任务类型
        prediction_time: 预测时间
        label_metadata: 标签元数据

        返回: 预测结果字典
        """
        if not self.initialized:
            raise RuntimeError("模型未初始化，请先调用initialize()")

        # 1. 编码查询图
        query_embedding = self.encode_graph(query_graph, prediction_time)

        # 2. 编码上下文图
        context_embeddings = []
        for ctx_graph in context_graphs:
            if ctx_graph is not None:
                ctx_embedding = self.encode_graph(ctx_graph, prediction_time)
                context_embeddings.append(ctx_embedding)
            else:
                # 使用零向量作为占位符
                context_embeddings.append(
                    torch.zeros_like(query_embedding)
                )

        context_embeddings = torch.stack(context_embeddings)

        # 3. 上下文学习
        predictions = self.icl_module(
            query_embeddings=query_embedding.unsqueeze(0),
            context_embeddings=context_embeddings.unsqueeze(0),
            context_labels=context_labels.unsqueeze(0),
            task_type=task_type,
            label_metadata=label_metadata
        )

        return predictions

    def encode_graph(self,
                     graph: HeterogeneousGraph,
                     reference_time: Optional[datetime] = None) -> torch.Tensor:
        """
        编码整个图为固定维度的表示

        graph: 异构图
        reference_time: 参考时间（用于时间编码）

        返回: (hidden_dim,) 图嵌入
        """
        # 1. 提取并编码节点特征
        node_features_by_table = self._encode_node_features(graph)

        # 2. 表级编码
        table_representations = self.table_encoder(
            tables_data=node_features_by_table,
            table_metadata=self._get_table_metadata(graph)
        )

        # 3. 准备图结构数据
        node_features_list = []
        node_types_list = []
        node_offsets = {}
        current_offset = 0

        # 按表顺序组织节点
        for table_idx, (table_name, table_nodes) in enumerate(graph.nodes.items()):
            if table_name in table_representations:
                table_features = table_representations[table_name]
                node_features_list.append(table_features)

                # 节点类型
                node_types = torch.full(
                    (len(table_nodes),),
                    table_idx,
                    dtype=torch.long
                )
                node_types_list.append(node_types)

                # 记录偏移
                node_offsets[table_name] = current_offset
                current_offset += len(table_nodes)

        if not node_features_list:
            # 空图
            return torch.zeros(self.config.hidden_dim)

        # 拼接所有节点
        all_node_features = torch.cat(node_features_list, dim=0)
        all_node_types = torch.cat(node_types_list, dim=0)

        # 4. 位置编码
        positional_info = self._prepare_positional_info(
            graph, all_node_types, reference_time
        )

        if positional_info:
            positional_encoding = self.positional_encoder(positional_info)
            if positional_encoding is not None:
                all_node_features = all_node_features + positional_encoding

        # 5. 构建边索引
        edge_index = self._build_edge_index(graph, node_offsets)

        # 6. 图Transformer
        graph_outputs = self.graph_transformer.transformer(
            node_features=all_node_features,
            edge_index=edge_index,
            node_types=all_node_types,
            center_node_idx=torch.tensor([0])  # 假设第一个节点是中心
        )

        # 7. 返回图嵌入
        return graph_outputs['graph_embedding'].squeeze(0)

    def _encode_node_features(self, graph: HeterogeneousGraph) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        编码图中所有节点的特征
        """
        encoded_features = {}

        for table_name, table_nodes in graph.nodes.items():
            if not table_nodes:
                continue

            # 收集该表所有节点的特征
            table_features = defaultdict(list)

            for node in table_nodes:
                for col_name, value in node.features.items():
                    table_features[col_name].append(value)

            # 编码每列
            encoded_columns = self.column_encoder(table_features)
            encoded_features[table_name] = encoded_columns

        return encoded_features

    def _get_table_metadata(self, graph: HeterogeneousGraph) -> Dict[str, Dict[str, Any]]:
        """
        获取表元数据
        """
        metadata = {}

        for table_idx, table_name in enumerate(graph.nodes.keys()):
            metadata[table_name] = {
                'table_type_id': table_idx,
                'column_order': list(graph.nodes[table_name][0].features.keys()) if graph.nodes[table_name] else []
            }

        return metadata

    def _prepare_positional_info(self,
                                 graph: HeterogeneousGraph,
                                 node_types: torch.Tensor,
                                 reference_time: Optional[datetime]) -> Dict[str, torch.Tensor]:
        """
        准备位置编码信息
        """
        info = {'node_types': node_types}

        # 跳数编码（简化：所有节点距离为0）
        if self.config.use_hop_encoding:
            info['hop_distances'] = torch.zeros_like(node_types)

        # 时间编码
        if self.config.use_time_encoding and reference_time:
            # 收集节点时间戳
            timestamps = []
            for table_nodes in graph.nodes.values():
                for node in table_nodes:
                    if node.timestamp:
                        timestamps.append(node.timestamp.timestamp())
                    else:
                        timestamps.append(reference_time.timestamp())

            info['node_timestamps'] = torch.tensor(timestamps)
            info['reference_timestamp'] = torch.tensor([reference_time.timestamp()])

        return info

    def _build_edge_index(self,
                          graph: HeterogeneousGraph,
                          node_offsets: Dict[str, int]) -> torch.Tensor:
        """
        构建全局边索引
        """
        all_edges = []

        # 创建节点ID到全局ID的映射
        node_id_to_global = {}
        for table_name, table_nodes in graph.nodes.items():
            offset = node_offsets.get(table_name, 0)
            for local_idx, node in enumerate(table_nodes):
                node_id_to_global[node.id] = offset + local_idx

        # 转换边
        for edge_type, edges in graph.edges.items():
            for edge in edges:
                if edge.source in node_id_to_global and edge.target in node_id_to_global:
                    global_src = node_id_to_global[edge.source]
                    global_tgt = node_id_to_global[edge.target]
                    all_edges.append([global_src, global_tgt])

        if all_edges:
            return torch.tensor(all_edges, dtype=torch.long).t()
        else:
            return torch.empty((2, 0), dtype=torch.long)

    def predict(self,
                database,
                entity_id: int,
                prediction_time: datetime,
                task_config: Dict[str, Any],
                context_generator: Optional[ContextGenerator] = None) -> Dict[str, Any]:
        """
        端到端预测接口

        database: 数据库对象
        entity_id: 要预测的实体ID
        prediction_time: 预测时间
        task_config: 任务配置
        context_generator: 上下文生成器

        返回: 预测结果
        """
        # 1. 构建查询子图
        graph_builder = GraphBuilder(database)
        full_graph = graph_builder.build_full_graph()

        backward_sampler = BackwardLookingSampler(full_graph, self.config.sampler_config)
        query_subgraph, _ = backward_sampler.sample_subgraph(
            center_node_id=entity_id,
            timestamp=prediction_time,
            num_hops=self.config.num_hops
        )

        # 2. 生成上下文
        if context_generator is None:
            forward_sampler = ForwardLookingSampler(full_graph, self.config.sampler_config)
            context_generator = ContextGenerator(
                self.config, full_graph, backward_sampler, forward_sampler
            )

        context_data = context_generator.generate_context_for_entity(
            entity_id=entity_id,
            prediction_time=prediction_time,
            task_config=task_config,
            num_context=self.config.num_context_examples
        )

        # 3. 执行预测
        predictions = self.forward(
            query_graph=query_subgraph,
            context_graphs=context_data['subgraphs'],
            context_labels=context_data['labels'],
            task_type=task_config['task_type'],
            prediction_time=prediction_time,
            label_metadata=task_config.get('label_metadata')
        )

        return predictions

    def explain(self,
                query_graph: HeterogeneousGraph,
                predictions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        生成预测解释
        """
        # 这里应该实现完整的解释功能
        # 包括特征重要性、注意力权重等
        return {
            'feature_importance': {},
            'attention_weights': {},
            'influential_nodes': []
        }