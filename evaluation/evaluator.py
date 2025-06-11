"""
评估器
用于评估模型性能
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from collections import defaultdict
import json
import pandas as pd
from datetime import datetime

from config.model_config import KumoRFMConfig
from .metrics import Metrics
from models.kumo_rfm import KumoRFM
from data.graph_builder import HeterogeneousGraph

logger = logging.getLogger(__name__)


class Evaluator:
    """
    模型评估器
    支持多种任务类型的评估
    """

    def __init__(self, config: KumoRFMConfig):
        self.config = config
        self.metrics = Metrics()

    def compute_metrics(self,
                        predictions: torch.Tensor,
                        targets: torch.Tensor,
                        task_type: str,
                        additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        计算指标

        predictions: 预测结果
        targets: 真实标签
        task_type: 任务类型
        additional_info: 额外信息（如时间戳等）
        """
        # 转换为numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        metrics_dict = {}

        if task_type == 'classification' or task_type == 'exists':
            # 分类任务
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                # 多分类
                y_pred = predictions.argmax(axis=1)
                y_prob = torch.softmax(torch.tensor(predictions), dim=1).numpy()
            else:
                # 二分类
                y_prob = torch.sigmoid(torch.tensor(predictions)).numpy()
                y_pred = (y_prob > 0.5).astype(int)
                if y_prob.ndim == 1:
                    y_prob = np.stack([1 - y_prob, y_prob], axis=1)

            metrics_dict.update(self.metrics.classification_metrics(
                targets.astype(int), y_pred, y_prob
            ))

        elif task_type in ['regression', 'count', 'sum']:
            # 回归任务
            metrics_dict.update(self.metrics.regression_metrics(targets, predictions))

        elif task_type in ['link_prediction', 'next', 'recommendation']:
            # 排序任务
            if additional_info and 'relevance_scores' in additional_info:
                # 有相关性分数
                relevance = additional_info['relevance_scores']
            else:
                # 使用二进制标签
                relevance = targets

            metrics_dict.update(self.metrics.ranking_metrics(relevance, predictions))

        # 添加时序指标（如果有时间戳）
        if additional_info and 'timestamps' in additional_info:
            temporal_metrics = self.metrics.temporal_metrics(
                targets, predictions, additional_info['timestamps']
            )
            metrics_dict.update(temporal_metrics)

        return metrics_dict

    def evaluate_model(self,
                       model: KumoRFM,
                       test_data: List[Dict[str, Any]],
                       task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估模型在测试集上的性能

        model: KumoRFM模型
        test_data: 测试数据列表
        task_config: 任务配置

        返回: 评估结果字典
        """
        model.eval()

        all_predictions = []
        all_targets = []
        all_metadata = []

        with torch.no_grad():
            for sample in test_data:
                try:
                    # 执行预测
                    predictions = model(
                        query_graph=sample['query_graph'],
                        context_graphs=sample['context_graphs'],
                        context_labels=sample['context_labels'],
                        task_type=task_config['task_type'],
                        prediction_time=sample.get('prediction_time'),
                        label_metadata=task_config.get('label_metadata')
                    )

                    # 提取预测值
                    if 'predictions' in predictions:
                        pred_value = predictions['predictions']
                    elif 'logits' in predictions:
                        pred_value = predictions['logits']
                    else:
                        pred_value = predictions.get('scores', torch.zeros(1))

                    all_predictions.append(pred_value)
                    all_targets.append(sample['target'])

                    # 收集元数据
                    metadata = {
                        'entity_id': sample.get('entity_id'),
                        'timestamp': sample.get('prediction_time')
                    }
                    all_metadata.append(metadata)

                except Exception as e:
                    logger.warning(f"评估样本失败: {e}")
                    continue

        if not all_predictions:
            logger.error("没有成功的预测")
            return {}

        # 合并预测结果
        if isinstance(all_predictions[0], torch.Tensor):
            predictions = torch.cat([p.unsqueeze(0) if p.dim() == 0 else p for p in all_predictions])
        else:
            predictions = torch.tensor(all_predictions)

        targets = torch.tensor(all_targets)

        # 计算指标
        metrics = self.compute_metrics(
            predictions=predictions,
            targets=targets,
            task_type=task_config['task_type'],
            additional_info={'metadata': all_metadata}
        )

        # 生成详细报告
        report = self.generate_report(
            metrics=metrics,
            predictions=predictions,
            targets=targets,
            metadata=all_metadata,
            task_config=task_config
        )

        return report

    def generate_report(self,
                        metrics: Dict[str, float],
                        predictions: torch.Tensor,
                        targets: torch.Tensor,
                        metadata: List[Dict[str, Any]],
                        task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成详细的评估报告
        """
        report = {
            'metrics': metrics,
            'task_config': task_config,
            'evaluation_time': datetime.now().isoformat(),
            'num_samples': len(predictions)
        }

        # 添加预测分析
        predictions_np = predictions.numpy()
        targets_np = targets.numpy()

        # 错误分析
        if task_config['task_type'] in ['classification', 'exists']:
            # 分类错误分析
            if predictions_np.ndim > 1:
                pred_classes = predictions_np.argmax(axis=1)
            else:
                pred_classes = (predictions_np > 0.5).astype(int)

            errors = pred_classes != targets_np
            error_indices = np.where(errors)[0]

            report['error_analysis'] = {
                'error_rate': errors.mean(),
                'num_errors': errors.sum(),
                'error_indices': error_indices.tolist()[:100]  # 最多100个
            }

            # 按类别分析
            class_performance = {}
            for class_id in np.unique(targets_np):
                class_mask = targets_np == class_id
                if class_mask.sum() > 0:
                    class_acc = (pred_classes[class_mask] == class_id).mean()
                    class_performance[f'class_{class_id}'] = {
                        'accuracy': float(class_acc),
                        'support': int(class_mask.sum())
                    }

            report['class_performance'] = class_performance

        elif task_config['task_type'] in ['regression', 'count', 'sum']:
            # 回归错误分析
            errors = np.abs(predictions_np - targets_np)

            report['error_analysis'] = {
                'mean_error': float(errors.mean()),
                'median_error': float(np.median(errors)),
                'max_error': float(errors.max()),
                'error_percentiles': {
                    'p50': float(np.percentile(errors, 50)),
                    'p90': float(np.percentile(errors, 90)),
                    'p95': float(np.percentile(errors, 95)),
                    'p99': float(np.percentile(errors, 99))
                }
            }

            # 找出最大误差的样本
            worst_indices = np.argsort(errors)[-10:][::-1]
            report['worst_predictions'] = [
                {
                    'index': int(idx),
                    'predicted': float(predictions_np[idx]),
                    'actual': float(targets_np[idx]),
                    'error': float(errors[idx]),
                    'metadata': metadata[idx] if idx < len(metadata) else {}
                }
                for idx in worst_indices
            ]

        # 时间分析（如果有时间戳）
        timestamps = [m.get('timestamp') for m in metadata if m.get('timestamp')]
        if timestamps:
            report['temporal_analysis'] = self._analyze_temporal_performance(
                predictions_np, targets_np, timestamps, task_config['task_type']
            )

        return report

    def _analyze_temporal_performance(self,
                                      predictions: np.ndarray,
                                      targets: np.ndarray,
                                      timestamps: List[datetime],
                                      task_type: str) -> Dict[str, Any]:
        """
        分析时序性能
        """
        # 转换时间戳为数组
        ts_array = np.array([ts.timestamp() if isinstance(ts, datetime) else ts for ts in timestamps])

        # 按时间排序
        sorted_indices = np.argsort(ts_array)

        # 将数据分成时间段
        num_buckets = min(10, len(timestamps) // 10)
        bucket_size = len(timestamps) // num_buckets

        temporal_performance = []

        for i in range(num_buckets):
            start_idx = i * bucket_size
            end_idx = (i + 1) * bucket_size if i < num_buckets - 1 else len(timestamps)

            bucket_indices = sorted_indices[start_idx:end_idx]
            bucket_pred = predictions[bucket_indices]
            bucket_true = targets[bucket_indices]

            # 计算桶内指标
            if task_type in ['classification', 'exists']:
                if bucket_pred.ndim > 1:
                    bucket_pred_class = bucket_pred.argmax(axis=1)
                else:
                    bucket_pred_class = (bucket_pred > 0.5).astype(int)

                accuracy = (bucket_pred_class == bucket_true).mean()
                metric_value = accuracy
                metric_name = 'accuracy'
            else:
                mae = np.abs(bucket_pred - bucket_true).mean()
                metric_value = mae
                metric_name = 'mae'

            # 时间范围
            bucket_timestamps = ts_array[bucket_indices]

            temporal_performance.append({
                'bucket': i,
                'start_time': datetime.fromtimestamp(bucket_timestamps.min()).isoformat(),
                'end_time': datetime.fromtimestamp(bucket_timestamps.max()).isoformat(),
                'num_samples': len(bucket_indices),
                metric_name: float(metric_value)
            })

        return {
            'buckets': temporal_performance,
            'trend': self._calculate_trend([b[metric_name] for b in temporal_performance])
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """
        计算趋势
        """
        if len(values) < 2:
            return 'stable'

        # 简单线性回归
        x = np.arange(len(values))
        y = np.array(values)

        # 计算斜率
        slope = np.polyfit(x, y, 1)[0]

        # 判断趋势
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'improving' if values[0] > values[-1] else 'declining'  # 考虑指标方向
        else:
            return 'declining' if values[0] > values[-1] else 'improving'

    def compare_models(self,
                       models: Dict[str, KumoRFM],
                       test_data: List[Dict[str, Any]],
                       task_config: Dict[str, Any]) -> pd.DataFrame:
        """
        比较多个模型的性能

        models: {模型名: 模型对象}
        test_data: 测试数据
        task_config: 任务配置

        返回: 比较结果DataFrame
        """
        results = {}

        for model_name, model in models.items():
            logger.info(f"评估模型: {model_name}")

            report = self.evaluate_model(model, test_data, task_config)

            # 提取主要指标
            metrics = report.get('metrics', {})
            results[model_name] = metrics

        # 转换为DataFrame
        df = pd.DataFrame(results).T

        # 添加排名
        for col in df.columns:
            if 'loss' in col or 'error' in col or 'mae' in col or 'mse' in col:
                # 越小越好的指标
                df[f'{col}_rank'] = df[col].rank()
            else:
                # 越大越好的指标
                df[f'{col}_rank'] = df[col].rank(ascending=False)

        return df

    def save_report(self, report: Dict[str, Any], filepath: str):
        """保存评估报告"""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"评估报告已保存到: {filepath}")

    def load_report(self, filepath: str) -> Dict[str, Any]:
        """加载评估报告"""
        with open(filepath, 'r') as f:
            report = json.load(f)

        return report