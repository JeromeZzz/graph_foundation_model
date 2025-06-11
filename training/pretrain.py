"""
KumoRFM预训练脚本
在多个数据集上进行预训练
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
from models.icl.context_generator import ContextGenerator
from .loss import ICLLoss
from utils.training_utils import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


class PretrainDataset(Dataset):
    """
    预训练数据集
    动态生成任务和上下文
    """

    def __init__(self,
                 databases: List[RelationalDatabase],
                 config: KumoRFMConfig,
                 num_samples_per_epoch: int = 10000):
        self.databases = databases
        self.config = config
        self.num_samples = num_samples_per_epoch

        # 为每个数据库构建图和采样器
        self.graphs = []
        self.samplers = []
        self.context_generators = []

        for db in databases:
            graph_builder = GraphBuilder(db)
            graph = graph_builder.build_full_graph()
            self.graphs.append(graph)

            backward_sampler = BackwardLookingSampler(graph, config.sampler_config)
            forward_sampler = ForwardLookingSampler(graph, config.sampler_config)
            self.samplers.append((backward_sampler, forward_sampler))

            context_gen = ContextGenerator(
                config, graph, backward_sampler, forward_sampler
            )
            self.context_generators.append(context_gen)

        # 预定义任务类型
        self.task_templates = [
            {
                'task_type': 'count',
                'aggregation_window': 7,
                'context_strategy': 'temporal_proximity'
            },
            {
                'task_type': 'sum',
                'aggregation_window': 30,
                'context_strategy': 'temporal_proximity'
            },
            {
                'task_type': 'exists',
                'aggregation_window': 1,
                'context_strategy': 'mixed'
            },
            {
                'task_type': 'next',
                'aggregation_window': 7,
                'context_strategy': 'structural_similarity'
            }
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机选择数据库
        db_idx = np.random.randint(0, len(self.databases))
        graph = self.graphs[db_idx]
        backward_sampler, forward_sampler = self.samplers[db_idx]
        context_gen = self.context_generators[db_idx]

        # 随机选择实体和时间
        all_nodes = []
        for table_nodes in graph.nodes.values():
            all_nodes.extend(table_nodes)

        # 过滤有时间戳的节点
        temporal_nodes = [n for n in all_nodes if n.timestamp]
        if not temporal_nodes:
            # 如果没有时序节点，返回空样本
            return self._get_empty_sample()

        # 随机选择节点
        node = np.random.choice(temporal_nodes)
        entity_id = node.id

        # 随机选择预测时间（节点时间之后）
        base_time = node.timestamp
        future_offset = np.random.randint(1, 90)  # 1-90天后
        prediction_time = base_time + timedelta(days=future_offset)

        # 随机选择任务
        task_config = np.random.choice(self.task_templates).copy()

        try:
            # 生成查询子图
            query_subgraph, _ = backward_sampler.sample_subgraph(
                center_node_id=entity_id,
                timestamp=prediction_time,
                num_hops=self.config.num_hops
            )

            # 生成真实标签
            true_label = forward_sampler.generate_label(
                node_id=entity_id,
                prediction_time=prediction_time,
                **task_config
            )

            if true_label is None:
                return self._get_empty_sample()

            # 生成上下文
            context_data = context_gen.generate_context_for_entity(
                entity_id=entity_id,
                prediction_time=prediction_time,
                task_config=task_config,
                num_context=self.config.num_context_examples
            )

            return {
                'query_graph': query_subgraph,
                'context_graphs': context_data['subgraphs'],
                'context_labels': context_data['labels'],
                'context_mask': context_data['mask'],
                'true_label': torch.tensor(true_label, dtype=torch.float32),
                'task_type': task_config['task_type'],
                'prediction_time': prediction_time,
                'db_idx': db_idx
            }

        except Exception as e:
            logger.warning(f"采样失败: {e}")
            return self._get_empty_sample()

    def _get_empty_sample(self):
        """返回空样本"""
        return {
            'query_graph': None,
            'context_graphs': [],
            'context_labels': torch.zeros(self.config.num_context_examples),
            'context_mask': torch.zeros(self.config.num_context_examples, dtype=torch.bool),
            'true_label': torch.tensor(0.0),
            'task_type': 'count',
            'prediction_time': datetime.now(),
            'db_idx': 0
        }


class PreTrainer:
    """
    预训练器
    """

    def __init__(self,
                 config: KumoRFMConfig,
                 model: KumoRFM,
                 train_databases: List[RelationalDatabase],
                 val_databases: Optional[List[RelationalDatabase]] = None,
                 use_wandb: bool = True):
        self.config = config
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 移动模型到设备
        self.model.to(self.device)

        # 数据集
        self.train_dataset = PretrainDataset(train_databases, config)
        if val_databases:
            self.val_dataset = PretrainDataset(val_databases, config, num_samples_per_epoch=1000)
        else:
            self.val_dataset = None

        # 数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn
        )

        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=self._collate_fn
            )

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100  # epochs
        )

        # 损失函数
        self.criterion = ICLLoss(config)

        # 工具
        self.early_stopping = EarlyStopping(patience=10)
        self.checkpoint = ModelCheckpoint(save_dir='checkpoints')

        # Wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="kumo-rfm-pretrain", config=config.__dict__)

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        批处理整理函数
        处理可变长度的图数据
        """
        # 过滤空样本
        valid_samples = [s for s in batch if s['query_graph'] is not None]

        if not valid_samples:
            # 返回空批次
            return None

        # 整理数据
        collated = {
            'query_graphs': [s['query_graph'] for s in valid_samples],
            'context_graphs': [s['context_graphs'] for s in valid_samples],
            'context_labels': torch.stack([s['context_labels'] for s in valid_samples]),
            'context_masks': torch.stack([s['context_mask'] for s in valid_samples]),
            'true_labels': torch.stack([s['true_label'] for s in valid_samples]),
            'task_types': [s['task_type'] for s in valid_samples],
            'prediction_times': [s['prediction_time'] for s in valid_samples],
            'db_indices': torch.tensor([s['db_idx'] for s in valid_samples])
        }

        return collated

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue

            # 清零梯度
            self.optimizer.zero_grad()

            # 批处理中的每个样本
            batch_loss = 0
            valid_count = 0

            for i in range(len(batch['query_graphs'])):
                try:
                    # 前向传播
                    predictions = self.model(
                        query_graph=batch['query_graphs'][i],
                        context_graphs=batch['context_graphs'][i],
                        context_labels=batch['context_labels'][i],
                        task_type=batch['task_types'][i],
                        prediction_time=batch['prediction_times'][i]
                    )

                    # 计算损失
                    loss = self.criterion(
                        predictions=predictions,
                        targets=batch['true_labels'][i:i + 1],
                        task_type=batch['task_types'][i]
                    )

                    batch_loss += loss
                    valid_count += 1

                except Exception as e:
                    logger.warning(f"样本处理失败: {e}")
                    continue

            if valid_count > 0:
                # 平均损失
                batch_loss = batch_loss / valid_count

                # 反向传播
                batch_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # 更新参数
                self.optimizer.step()

                # 累积损失
                total_loss += batch_loss.item() * valid_count
                total_samples += valid_count

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': batch_loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })

            # 梯度累积
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # 计算平均损失
        avg_loss = total_loss / max(total_samples, 1)

        return {'loss': avg_loss}

    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if batch is None:
                    continue

                batch_loss = 0
                valid_count = 0

                for i in range(len(batch['query_graphs'])):
                    try:
                        # 前向传播
                        predictions = self.model(
                            query_graph=batch['query_graphs'][i],
                            context_graphs=batch['context_graphs'][i],
                            context_labels=batch['context_labels'][i],
                            task_type=batch['task_types'][i],
                            prediction_time=batch['prediction_times'][i]
                        )

                        # 计算损失
                        loss = self.criterion(
                            predictions=predictions,
                            targets=batch['true_labels'][i:i + 1],
                            task_type=batch['task_types'][i]
                        )

                        batch_loss += loss
                        valid_count += 1

                    except Exception as e:
                        logger.warning(f"验证样本处理失败: {e}")
                        continue

                if valid_count > 0:
                    total_loss += batch_loss.item()
                    total_samples += valid_count

        avg_loss = total_loss / max(total_samples, 1)

        return {'val_loss': avg_loss}

    def train(self, num_epochs: int):
        """
        执行预训练
        """
        logger.info(f"开始预训练，共 {num_epochs} 个epoch")

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # 训练
            train_metrics = self.train_epoch()
            logger.info(f"训练指标: {train_metrics}")

            # 验证
            val_metrics = self.validate()
            logger.info(f"验证指标: {val_metrics}")

            # 更新学习率
            self.scheduler.step()

            # 记录到wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics.get('val_loss', 0),
                    'lr': self.optimizer.param_groups[0]['lr']
                })

            # 早停检查
            if 'val_loss' in val_metrics:
                self.early_stopping(val_metrics['val_loss'])
                if self.early_stopping.early_stop:
                    logger.info("早停触发，停止训练")
                    break

            # 保存检查点
            self.checkpoint.save(
                self.model,
                self.optimizer,
                epoch + 1,
                train_metrics['loss'],
                val_metrics.get('val_loss')
            )

        logger.info("预训练完成")

        # 保存最终模型
        torch.save(
            self.model.state_dict(),
            os.path.join(self.checkpoint.save_dir, 'final_model.pt')
        )