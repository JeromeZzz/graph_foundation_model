"""
KumoRFM微调脚本
针对特定任务进行微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import wandb
from datetime import datetime
import os

from config.model_config import KumoRFMConfig
from models.kumo_rfm import KumoRFM
from data.database import RelationalDatabase
from data.graph_builder import GraphBuilder
from data.samplers import BackwardLookingSampler, ForwardLookingSampler
from pql.parser import PQLParser
from .loss import get_loss_function
from evaluation.evaluator import Evaluator
from utils.training_utils import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


class FinetuneDataset(Dataset):
    """
    微调数据集
    针对特定任务生成训练数据
    """

    def __init__(self,
                 database: RelationalDatabase,
                 pql_query: str,
                 train_entities: List[Tuple[Any, datetime]],
                 config: KumoRFMConfig):
        self.database = database
        self.config = config

        # 解析PQL查询
        parser = PQLParser()
        self.pql_query = parser.parse(pql_query)
        self.task_config = parser.to_task_config(self.pql_query)

        # 构建图
        graph_builder = GraphBuilder(database)
        self.graph = graph_builder.build_full_graph()

        # 采样器
        self.backward_sampler = BackwardLookingSampler(self.graph, config.sampler_config)
        self.forward_sampler = ForwardLookingSampler(self.graph, config.sampler_config)

        # 训练实体
        self.train_entities = train_entities

        # 实体ID映射
        self._build_entity_mapping()

    def _build_entity_mapping(self):
        """构建实体值到图节点ID的映射"""
        self.entity_to_node_id = {}

        entity_table = self.pql_query.entity_table
        entity_column = self.pql_query.entity_column

        # 查找对应的节点
        for node in self.graph.nodes.get(entity_table, []):
            if entity_column in node.features:
                entity_value = node.features[entity_column]
                self.entity_to_node_id[entity_value] = node.id

    def __len__(self):
        return len(self.train_entities)

    def __getitem__(self, idx):
        entity_value, prediction_time = self.train_entities[idx]

        # 获取节点ID
        node_id = self.entity_to_node_id.get(entity_value)
        if node_id is None:
            raise ValueError(f"找不到实体 {entity_value} 的节点")

        try:
            # 生成查询子图
            query_subgraph, _ = self.backward_sampler.sample_subgraph(
                center_node_id=node_id,
                timestamp=prediction_time,
                num_hops=self.config.num_hops
            )

            # 生成标签
            label = self.forward_sampler.generate_label(
                node_id=node_id,
                prediction_time=prediction_time,
                task_type=self.task_config['task_type'],
                aggregation_window=self.task_config['time_window']['end'],
                target_column=self.task_config.get('target_column')
            )

            if label is None:
                label = 0  # 默认标签

            return {
                'query_graph': query_subgraph,
                'label': torch.tensor(label, dtype=torch.float32),
                'entity_value': entity_value,
                'prediction_time': prediction_time
            }

        except Exception as e:
            logger.error(f"生成数据失败: {e}")
            raise


class FineTuner:
    """
    微调器
    """

    def __init__(self,
                 config: KumoRFMConfig,
                 model: KumoRFM,
                 database: RelationalDatabase,
                 pql_query: str,
                 save_dir: str = "finetuned_models"):
        self.config = config
        self.model = model
        self.database = database
        self.pql_query = pql_query
        self.save_dir = save_dir

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 解析任务
        parser = PQLParser()
        self.parsed_query = parser.parse(pql_query)
        self.task_config = parser.to_task_config(self.parsed_query)

        # 替换ICL头为任务特定头
        self._replace_task_head()

        # 优化器（只优化新的任务头）
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate * 0.1  # 使用更小的学习率
        )

        # 损失函数
        self.criterion = get_loss_function(config, self.task_config['task_type'])

        # 评估器
        self.evaluator = Evaluator(config)

        # 工具
        self.early_stopping = EarlyStopping(patience=5)
        self.checkpoint = ModelCheckpoint(save_dir)

        logger.info(f"微调器初始化完成，任务: {pql_query}")

    def _replace_task_head(self):
        """替换为任务特定的预测头"""
        task_type = self.task_config['task_type']

        if task_type == 'classification':
            num_classes = self.task_config.get('num_classes', 2)
            self.task_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.hidden_dropout),
                nn.Linear(self.config.hidden_dim, num_classes)
            )

        elif task_type == 'regression':
            self.task_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.config.hidden_dropout),
                nn.Linear(self.config.hidden_dim // 2, 1)
            )

        elif task_type == 'link_prediction':
            # 链接预测使用双塔结构
            self.user_tower = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
            self.item_tower = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
            self.task_head = nn.ModuleDict({
                'user': self.user_tower,
                'item': self.item_tower
            })

        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

        # 移动到设备
        if hasattr(self, 'task_head'):
            self.task_head.to(self.device)

    def prepare_data(self,
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        准备训练、验证和测试数据
        """
        # 获取所有实体和时间戳
        all_entities = self._get_all_entities()

        # 随机划分
        np.random.shuffle(all_entities)
        n = len(all_entities)

        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train_entities = all_entities[:train_size]
        val_entities = all_entities[train_size:train_size + val_size]
        test_entities = all_entities[train_size + val_size:]

        # 创建数据集
        train_dataset = FinetuneDataset(
            self.database, self.pql_query, train_entities, self.config
        )
        val_dataset = FinetuneDataset(
            self.database, self.pql_query, val_entities, self.config
        )
        test_dataset = FinetuneDataset(
            self.database, self.pql_query, test_entities, self.config
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate_fn
        )

        return train_loader, val_loader, test_loader

    def _get_all_entities(self) -> List[Tuple[Any, datetime]]:
        """获取所有符合条件的实体"""
        entities = []

        # 从数据库获取实体表
        entity_table = self.database.get_table(self.parsed_query.entity_table)
        entity_column = self.parsed_query.entity_column

        # 如果指定了实体值，使用指定的
        if self.parsed_query.entity_values:
            entity_values = self.parsed_query.entity_values
        else:
            # 否则使用所有实体
            entity_values = entity_table.data[entity_column].unique()

        # 生成时间戳
        # 这里简化处理，使用固定的时间范围
        base_time = datetime(2023, 1, 1)
        for i in range(100):  # 100个时间点
            timestamp = base_time + timedelta(days=i * 7)  # 每周一个点
            for entity_value in entity_values:
                entities.append((entity_value, timestamp))

        return entities

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """批处理整理函数"""
        return {
            'query_graphs': [item['query_graph'] for item in batch],
            'labels': torch.stack([item['label'] for item in batch]),
            'entity_values': [item['entity_value'] for item in batch],
            'prediction_times': [item['prediction_time'] for item in batch]
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0
        total_samples = 0

        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()

            batch_loss = 0

            for i in range(len(batch['query_graphs'])):
                # 编码查询图
                query_embedding = self.model.encode_graph(
                    batch['query_graphs'][i],
                    batch['prediction_times'][i]
                )

                # 应用任务头
                if self.task_config['task_type'] == 'link_prediction':
                    # 链接预测特殊处理
                    predictions = self.task_head['user'](query_embedding)
                else:
                    predictions = self.task_head(query_embedding)

                # 计算损失
                if predictions.dim() > 1 and predictions.shape[-1] == 1:
                    predictions = predictions.squeeze(-1)

                loss = self.criterion(
                    predictions.unsqueeze(0),
                    batch['labels'][i:i + 1]
                )

                batch_loss += loss

            # 平均损失
            batch_loss = batch_loss / len(batch['query_graphs'])
            batch_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            self.optimizer.step()

            total_loss += batch_loss.item() * len(batch['query_graphs'])
            total_samples += len(batch['query_graphs'])

        return {'loss': total_loss / total_samples}

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """评估"""
        self.model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                for i in range(len(batch['query_graphs'])):
                    # 编码查询图
                    query_embedding = self.model.encode_graph(
                        batch['query_graphs'][i],
                        batch['prediction_times'][i]
                    )

                    # 应用任务头
                    if self.task_config['task_type'] == 'link_prediction':
                        predictions = self.task_head['user'](query_embedding)
                    else:
                        predictions = self.task_head(query_embedding)

                    if predictions.dim() > 1 and predictions.shape[-1] == 1:
                        predictions = predictions.squeeze(-1)

                    all_predictions.append(predictions.cpu())
                    all_labels.append(batch['labels'][i].cpu())

        # 计算指标
        predictions = torch.stack(all_predictions)
        labels = torch.stack(all_labels)

        metrics = self.evaluator.compute_metrics(
            predictions, labels, self.task_config['task_type']
        )

        return metrics

    def finetune(self, num_epochs: int = 10):
        """
        执行微调
        """
        logger.info(f"开始微调，共 {num_epochs} 个epoch")

        # 准备数据
        train_loader, val_loader, test_loader = self.prepare_data()

        best_val_metric = None

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # 训练
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"训练损失: {train_metrics['loss']:.4f}")

            # 验证
            val_metrics = self.evaluate(val_loader)
            logger.info(f"验证指标: {val_metrics}")

            # 选择主要指标
            if self.task_config['task_type'] == 'classification':
                main_metric = val_metrics.get('accuracy', 0)
            elif self.task_config['task_type'] == 'regression':
                main_metric = -val_metrics.get('mae', float('inf'))
            else:
                main_metric = val_metrics.get('map@k', 0)

            # 保存最佳模型
            if best_val_metric is None or main_metric > best_val_metric:
                best_val_metric = main_metric
                self.save_model('best_model.pt')
                logger.info("保存最佳模型")

            # 早停检查
            self.early_stopping(-main_metric)
            if self.early_stopping.early_stop:
                logger.info("早停触发")
                break

        # 加载最佳模型并测试
        self.load_model('best_model.pt')
        test_metrics = self.evaluate(test_loader)
        logger.info(f"\n测试集指标: {test_metrics}")

        # 保存最终模型
        self.save_model('final_model.pt')

        return test_metrics

    def save_model(self, filename: str):
        """保存模型"""
        os.makedirs(self.save_dir, exist_ok=True)

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'task_head_state_dict': self.task_head.state_dict() if hasattr(self, 'task_head') else None,
            'task_config': self.task_config,
            'pql_query': self.pql_query
        }

        torch.save(save_dict, os.path.join(self.save_dir, filename))

    def load_model(self, filename: str):
        """加载模型"""
        checkpoint = torch.load(os.path.join(self.save_dir, filename))

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if checkpoint['task_head_state_dict'] and hasattr(self, 'task_head'):
            self.task_head.load_state_dict(checkpoint['task_head_state_dict'])