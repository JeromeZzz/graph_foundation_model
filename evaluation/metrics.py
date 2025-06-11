"""
评估指标
包括分类、回归和排序任务的各种指标
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    ndcg_score, mean_absolute_percentage_error
)
import warnings

warnings.filterwarnings('ignore')


class Metrics:
    """
    统一的指标计算类
    """

    @staticmethod
    def classification_metrics(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None,
                               average: str = 'macro') -> Dict[str, float]:
        """
        计算分类指标

        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选）
        average: 多分类平均方式
        """
        metrics = {}

        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # 处理二分类和多分类
        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) == 2

        if is_binary:
            # 二分类指标
            metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            metrics['f1'] = f1_score(y_true, y_pred, average='binary')

            if y_prob is not None:
                # 需要概率的指标
                if y_prob.ndim > 1:
                    y_prob_binary = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob[:, 0]
                else:
                    y_prob_binary = y_prob

                try:
                    metrics['auroc'] = roc_auc_score(y_true, y_prob_binary)
                    metrics['auprc'] = average_precision_score(y_true, y_prob_binary)
                except:
                    metrics['auroc'] = 0.0
                    metrics['auprc'] = 0.0

        else:
            # 多分类指标
            metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)

            # 每个类别的指标
            for i, class_label in enumerate(unique_classes):
                class_mask = y_true == class_label
                if class_mask.sum() > 0:
                    metrics[f'precision_class_{class_label}'] = precision_score(
                        y_true == class_label, y_pred == class_label, average='binary', zero_division=0
                    )
                    metrics[f'recall_class_{class_label}'] = recall_score(
                        y_true == class_label, y_pred == class_label, average='binary', zero_division=0
                    )

            if y_prob is not None and y_prob.shape[1] == len(unique_classes):
                # 多分类AUC
                try:
                    metrics['auroc'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average=average
                    )
                except:
                    metrics['auroc'] = 0.0

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        return metrics

    @staticmethod
    def regression_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算回归指标
        """
        metrics = {}

        # 基础指标
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)

        # MAPE（处理零值）
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
        else:
            metrics['mape'] = 0.0

        # 额外统计
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)

        # 分位数误差
        metrics['q25_error'] = np.percentile(np.abs(residuals), 25)
        metrics['q50_error'] = np.percentile(np.abs(residuals), 50)
        metrics['q75_error'] = np.percentile(np.abs(residuals), 75)
        metrics['q90_error'] = np.percentile(np.abs(residuals), 90)

        return metrics

    @staticmethod
    def ranking_metrics(y_true: Union[np.ndarray, List[np.ndarray]],
                        y_scores: Union[np.ndarray, List[np.ndarray]],
                        k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """
        计算排序指标

        y_true: 真实相关性分数或二进制标签
        y_scores: 预测分数
        k_values: top-k的k值列表
        """
        metrics = {}

        # 处理输入格式
        if isinstance(y_true, list):
            # 多个查询的情况
            all_metrics = []
            for true, scores in zip(y_true, y_scores):
                query_metrics = Metrics._ranking_metrics_single(true, scores, k_values)
                all_metrics.append(query_metrics)

            # 平均所有查询的指标
            for key in all_metrics[0].keys():
                metrics[key] = np.mean([m[key] for m in all_metrics])
        else:
            # 单个查询
            metrics = Metrics._ranking_metrics_single(y_true, y_scores, k_values)

        return metrics

    @staticmethod
    def _ranking_metrics_single(y_true: np.ndarray,
                                y_scores: np.ndarray,
                                k_values: List[int]) -> Dict[str, float]:
        """
        计算单个查询的排序指标
        """
        metrics = {}

        # 确保是numpy数组
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        # 按分数排序
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_true = y_true[sorted_indices]

        # 计算不同k值的指标
        for k in k_values:
            if k > len(y_true):
                k = len(y_true)

            # Precision@k
            if sorted_true[:k].sum() > 0:
                precision_k = sorted_true[:k].sum() / k
            else:
                precision_k = 0.0
            metrics[f'precision@{k}'] = precision_k

            # Recall@k
            if y_true.sum() > 0:
                recall_k = sorted_true[:k].sum() / y_true.sum()
            else:
                recall_k = 0.0
            metrics[f'recall@{k}'] = recall_k

            # Hit@k
            hit_k = 1.0 if sorted_true[:k].sum() > 0 else 0.0
            metrics[f'hit@{k}'] = hit_k

            # MAP@k
            map_k = Metrics._average_precision_at_k(sorted_true, k)
            metrics[f'map@{k}'] = map_k

        # NDCG
        for k in k_values:
            if k > len(y_true):
                k = len(y_true)

            ndcg_k = Metrics._ndcg_at_k(y_true, y_scores, k)
            metrics[f'ndcg@{k}'] = ndcg_k

        # MRR（Mean Reciprocal Rank）
        mrr = Metrics._mean_reciprocal_rank(sorted_true)
        metrics['mrr'] = mrr

        return metrics

    @staticmethod
    def _average_precision_at_k(sorted_true: np.ndarray, k: int) -> float:
        """计算AP@k"""
        if k > len(sorted_true):
            k = len(sorted_true)

        sorted_true_k = sorted_true[:k]

        if sorted_true_k.sum() == 0:
            return 0.0

        precisions = []
        num_relevant = 0

        for i in range(k):
            if sorted_true_k[i] == 1:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precisions.append(precision_at_i)

        if not precisions:
            return 0.0

        return np.mean(precisions)

    @staticmethod
    def _ndcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        """计算NDCG@k"""
        try:
            # sklearn的ndcg_score需要2D输入
            y_true_2d = y_true.reshape(1, -1)
            y_scores_2d = y_scores.reshape(1, -1)

            return ndcg_score(y_true_2d, y_scores_2d, k=k)
        except:
            return 0.0

    @staticmethod
    def _mean_reciprocal_rank(sorted_true: np.ndarray) -> float:
        """计算MRR"""
        # 找到第一个相关项目的位置
        relevant_positions = np.where(sorted_true == 1)[0]

        if len(relevant_positions) > 0:
            first_relevant_position = relevant_positions[0]
            return 1.0 / (first_relevant_position + 1)
        else:
            return 0.0

    @staticmethod
    def diversity_metrics(recommendations: List[List[int]],
                          item_features: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算推荐多样性指标

        recommendations: 推荐列表的列表
        item_features: 物品特征矩阵（可选）
        """
        metrics = {}

        # 计算覆盖率
        all_items = set()
        for rec_list in recommendations:
            all_items.update(rec_list)

        if item_features is not None:
            total_items = item_features.shape[0]
        else:
            total_items = max(all_items) + 1 if all_items else 0

        metrics['coverage'] = len(all_items) / max(total_items, 1)

        # 计算平均内部多样性
        if item_features is not None:
            diversities = []
            for rec_list in recommendations:
                if len(rec_list) > 1:
                    # 计算推荐列表内的平均距离
                    features = item_features[rec_list]
                    distances = []

                    for i in range(len(rec_list)):
                        for j in range(i + 1, len(rec_list)):
                            dist = np.linalg.norm(features[i] - features[j])
                            distances.append(dist)

                    if distances:
                        diversities.append(np.mean(distances))

            if diversities:
                metrics['avg_diversity'] = np.mean(diversities)
            else:
                metrics['avg_diversity'] = 0.0

        # 计算基尼系数（衡量推荐的公平性）
        item_counts = {}
        for rec_list in recommendations:
            for item in rec_list:
                item_counts[item] = item_counts.get(item, 0) + 1

        if item_counts:
            counts = np.array(list(item_counts.values()))
            gini = Metrics._gini_coefficient(counts)
            metrics['gini'] = gini
        else:
            metrics['gini'] = 0.0

        return metrics

    @staticmethod
    def _gini_coefficient(values: np.ndarray) -> float:
        """计算基尼系数"""
        # 排序
        sorted_values = np.sort(values)
        n = len(values)

        # 计算累积和
        cumsum = np.cumsum(sorted_values)

        # 计算基尼系数
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n

    @staticmethod
    def temporal_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         timestamps: np.ndarray) -> Dict[str, float]:
        """
        计算时序相关的指标

        timestamps: 预测的时间戳
        """
        metrics = {}

        # 按时间排序
        sorted_indices = np.argsort(timestamps)
        y_true_sorted = y_true[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]
        timestamps_sorted = timestamps[sorted_indices]

        # 计算不同时间段的性能
        # 将时间分成若干段
        num_segments = 5
        segment_size = len(timestamps) // num_segments

        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(timestamps)

            segment_true = y_true_sorted[start_idx:end_idx]
            segment_pred = y_pred_sorted[start_idx:end_idx]

            # 根据数据类型计算指标
            if np.unique(segment_true).size <= 10:  # 分类
                segment_acc = accuracy_score(segment_true, segment_pred)
                metrics[f'accuracy_segment_{i}'] = segment_acc
            else:  # 回归
                segment_mae = mean_absolute_error(segment_true, segment_pred)
                metrics[f'mae_segment_{i}'] = segment_mae

        # 计算预测的时间稳定性
        # 使用滑动窗口计算方差
        window_size = max(10, len(timestamps) // 20)
        variances = []

        for i in range(len(y_pred_sorted) - window_size):
            window_pred = y_pred_sorted[i:i + window_size]
            variances.append(np.var(window_pred))

        if variances:
            metrics['prediction_stability'] = 1.0 / (1.0 + np.mean(variances))
        else:
            metrics['prediction_stability'] = 1.0

        return metrics