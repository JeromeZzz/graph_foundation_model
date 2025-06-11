"""
分析型解释器
基于统计分析的全局和局部解释方法
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AnalyticalExplainer:
    """
    分析型解释器
    通过统计分析提供模型预测的解释
    """

    def __init__(self):
        self.feature_importance_cache = {}

    def explain_global(self,
                       predictions: torch.Tensor,
                       features: Dict[str, torch.Tensor],
                       labels: torch.Tensor,
                       column_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        全局解释：分析各个特征对整体预测的影响

        predictions: 模型预测
        features: 特征字典 {列名: 特征张量}
        labels: 真实标签
        column_metadata: 列元数据

        返回: 全局解释结果
        """
        global_importance = {}

        for col_name, col_features in features.items():
            # 获取列的数据类型
            dtype = column_metadata.get(col_name, {}).get('dtype', 'numerical')

            if dtype == 'numerical':
                importance = self._compute_numerical_importance(
                    col_features, predictions, labels
                )
            elif dtype == 'categorical':
                importance = self._compute_categorical_importance(
                    col_features, predictions, labels
                )
            else:
                importance = self._compute_generic_importance(
                    col_features, predictions, labels
                )

            global_importance[col_name] = {
                'importance_score': importance,
                'dtype': dtype,
                'distribution': self._get_feature_distribution(col_features, dtype)
            }

        # 排序特征重要性
        sorted_importance = sorted(
            global_importance.items(),
            key=lambda x: x[1]['importance_score'],
            reverse=True
        )

        return {
            'feature_importance': dict(sorted_importance),
            'top_features': [k for k, _ in sorted_importance[:10]]
        }

    def explain_local(self,
                      prediction: torch.Tensor,
                      instance_features: Dict[str, torch.Tensor],
                      global_features: Dict[str, torch.Tensor],
                      column_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        局部解释：解释单个预测

        prediction: 单个预测
        instance_features: 实例特征
        global_features: 全局特征分布
        column_metadata: 列元数据

        返回: 局部解释结果
        """
        local_importance = {}

        for col_name, col_value in instance_features.items():
            # 计算该特征值在全局分布中的位置
            global_dist = global_features.get(col_name)
            if global_dist is None:
                continue

            dtype = column_metadata.get(col_name, {}).get('dtype', 'numerical')

            # 计算特征贡献
            if dtype == 'numerical':
                contribution = self._compute_numerical_contribution(
                    col_value, global_dist, prediction
                )
            elif dtype == 'categorical':
                contribution = self._compute_categorical_contribution(
                    col_value, global_dist, prediction
                )
            else:
                contribution = 0.0

            local_importance[col_name] = {
                'value': col_value.item() if torch.is_tensor(col_value) else col_value,
                'contribution': contribution,
                'percentile': self._compute_percentile(col_value, global_dist)
            }

        return {
            'local_importance': local_importance,
            'prediction_confidence': self._compute_confidence(prediction)
        }

    def compute_cohort_analysis(self,
                                features: Dict[str, torch.Tensor],
                                labels: torch.Tensor,
                                cohort_column: str,
                                num_cohorts: int = 5) -> Dict[str, Any]:
        """
        队列分析：按特征值分组分析预测性能

        features: 特征字典
        labels: 标签
        cohort_column: 用于分组的列
        num_cohorts: 队列数量

        返回: 队列分析结果
        """
        if cohort_column not in features:
            raise ValueError(f"列 {cohort_column} 不存在")

        cohort_feature = features[cohort_column]

        # 创建队列
        cohorts = self._create_cohorts(cohort_feature, num_cohorts)

        cohort_analysis = []
        for i, (min_val, max_val, indices) in enumerate(cohorts):
            cohort_labels = labels[indices]

            # 计算队列统计
            cohort_stats = {
                'cohort_id': i,
                'range': (min_val, max_val),
                'size': len(indices),
                'mean_label': cohort_labels.float().mean().item(),
                'std_label': cohort_labels.float().std().item() if len(indices) > 1 else 0
            }

            cohort_analysis.append(cohort_stats)

        # 计算队列间的方差作为重要性指标
        cohort_means = [c['mean_label'] for c in cohort_analysis]
        importance = np.var(cohort_means)

        return {
            'cohorts': cohort_analysis,
            'feature_importance': importance,
            'cohort_variance': np.var(cohort_means)
        }

    def _compute_numerical_importance(self,
                                      features: torch.Tensor,
                                      predictions: torch.Tensor,
                                      labels: torch.Tensor) -> float:
        """计算数值特征的重要性"""
        # 使用相关系数
        if features.dim() > 1:
            features = features.mean(dim=1)

        # 计算与预测的相关性
        pred_corr = torch.corrcoef(torch.stack([features, predictions.squeeze()]))[0, 1]

        # 计算与标签的相关性
        label_corr = torch.corrcoef(torch.stack([features, labels.float()]))[0, 1]

        # 综合重要性
        importance = abs(pred_corr.item()) * 0.5 + abs(label_corr.item()) * 0.5

        return importance

    def _compute_categorical_importance(self,
                                        features: torch.Tensor,
                                        predictions: torch.Tensor,
                                        labels: torch.Tensor) -> float:
        """计算分类特征的重要性"""
        # 计算每个类别的平均预测值
        unique_values = torch.unique(features)
        category_means = []

        for val in unique_values:
            mask = features == val
            if mask.any():
                mean_pred = predictions[mask].mean()
                category_means.append(mean_pred)

        # 类别间方差作为重要性
        if category_means:
            importance = torch.var(torch.stack(category_means)).item()
        else:
            importance = 0.0

        return importance

    def _compute_generic_importance(self,
                                    features: torch.Tensor,
                                    predictions: torch.Tensor,
                                    labels: torch.Tensor) -> float:
        """计算通用特征的重要性"""
        # 简单使用预测值的标准差
        return predictions.std().item()

    def _get_feature_distribution(self,
                                  features: torch.Tensor,
                                  dtype: str) -> Dict[str, Any]:
        """获取特征分布"""
        if dtype == 'numerical':
            return {
                'mean': features.mean().item(),
                'std': features.std().item(),
                'min': features.min().item(),
                'max': features.max().item(),
                'quantiles': torch.quantile(features, torch.tensor([0.25, 0.5, 0.75])).tolist()
            }
        elif dtype == 'categorical':
            unique_values, counts = torch.unique(features, return_counts=True)
            return {
                'unique_values': unique_values.tolist(),
                'counts': counts.tolist(),
                'mode': unique_values[counts.argmax()].item()
            }
        else:
            return {}

    def _compute_numerical_contribution(self,
                                        value: torch.Tensor,
                                        global_dist: torch.Tensor,
                                        prediction: torch.Tensor) -> float:
        """计算数值特征的贡献"""
        # 计算z-score
        mean = global_dist.mean()
        std = global_dist.std()
        z_score = (value - mean) / (std + 1e-8)

        # 贡献与z-score和预测值相关
        contribution = z_score.item() * prediction.item()

        return contribution

    def _compute_categorical_contribution(self,
                                          value: torch.Tensor,
                                          global_dist: torch.Tensor,
                                          prediction: torch.Tensor) -> float:
        """计算分类特征的贡献"""
        # 计算该类别的频率
        unique_values, counts = torch.unique(global_dist, return_counts=True)

        # 找到当前值的索引
        mask = unique_values == value
        if mask.any():
            frequency = counts[mask].float() / counts.sum()
            # 稀有类别有更高的贡献
            contribution = (1 - frequency.item()) * prediction.item()
        else:
            contribution = 0.0

        return contribution

    def _compute_percentile(self,
                            value: torch.Tensor,
                            distribution: torch.Tensor) -> float:
        """计算值在分布中的百分位"""
        if distribution.numel() == 0:
            return 0.0

        # 计算小于等于该值的比例
        percentile = (distribution <= value).float().mean().item() * 100

        return percentile

    def _compute_confidence(self, prediction: torch.Tensor) -> float:
        """计算预测置信度"""
        # 对于分类任务，使用概率
        if prediction.dim() > 1:
            # 多分类
            probs = torch.softmax(prediction, dim=-1)
            confidence = probs.max().item()
        else:
            # 二分类或回归
            if prediction.min() >= 0 and prediction.max() <= 1:
                # 概率值
                confidence = max(prediction.item(), 1 - prediction.item())
            else:
                # 回归值，使用归一化
                confidence = 1.0  # 回归任务默认置信度

        return confidence

    def _create_cohorts(self,
                        features: torch.Tensor,
                        num_cohorts: int) -> List[Tuple[float, float, torch.Tensor]]:
        """创建特征队列"""
        # 排序特征值
        sorted_features, sorted_indices = torch.sort(features)

        # 创建等大小的队列
        cohort_size = len(features) // num_cohorts
        cohorts = []

        for i in range(num_cohorts):
            start_idx = i * cohort_size
            if i == num_cohorts - 1:
                # 最后一个队列包含所有剩余元素
                end_idx = len(features)
            else:
                end_idx = (i + 1) * cohort_size

            cohort_indices = sorted_indices[start_idx:end_idx]
            min_val = sorted_features[start_idx].item()
            max_val = sorted_features[end_idx - 1].item()

            cohorts.append((min_val, max_val, cohort_indices))

        return cohorts

    def generate_text_explanation(self,
                                  global_explanation: Dict[str, Any],
                                  local_explanation: Optional[Dict[str, Any]] = None,
                                  task_type: str = 'classification') -> str:
        """生成文本解释"""
        explanation_parts = []

        # 全局解释
        if global_explanation:
            top_features = global_explanation.get('top_features', [])[:5]
            explanation_parts.append("主要影响因素：")

            for i, feature in enumerate(top_features, 1):
                importance = global_explanation['feature_importance'][feature]
                explanation_parts.append(
                    f"{i}. {feature} (重要性: {importance['importance_score']:.3f})"
                )

        # 局部解释
        if local_explanation:
            explanation_parts.append("\n具体到此预测：")

            local_imp = local_explanation['local_importance']
            # 按贡献排序
            sorted_features = sorted(
                local_imp.items(),
                key=lambda x: abs(x[1]['contribution']),
                reverse=True
            )[:3]

            for feature, info in sorted_features:
                explanation_parts.append(
                    f"- {feature}: 值={info['value']}, "
                    f"贡献={info['contribution']:.3f}, "
                    f"百分位={info['percentile']:.1f}%"
                )

        return "\n".join(explanation_parts)
