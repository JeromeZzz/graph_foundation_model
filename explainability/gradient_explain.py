"""
梯度型解释器
基于梯度的解释方法，包括Saliency、Integrated Gradients等
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)


class GradientExplainer:
    """
    梯度型解释器
    使用梯度方法解释模型预测
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()  # 确保在评估模式

    def explain_saliency(self,
                        query_graph,
                        context_graphs: List,
                        context_labels: torch.Tensor,
                        task_type: str,
                        target_class: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        使用Saliency方法计算特征重要性
        
        query_graph: 查询图
        context_graphs: 上下文图列表
        context_labels: 上下文标签
        task_type: 任务类型
        target_class: 目标类别（分类任务）
        
        返回: 梯度字典
        """
        # 确保需要梯度
        self.model.zero_grad()
        
        # 启用梯度计算
        query_graph_with_grad = self._enable_gradients(query_graph)
        
        # 前向传播
        outputs = self.model(
            query_graph=query_graph_with_grad,
            context_graphs=context_graphs,
            context_labels=context_labels,
            task_type=task_type
        )
        
        # 选择目标输出
        if task_type == 'classification':
            if 'logits' in outputs:
                output = outputs['logits']
                if target_class is not None:
                    output = output[:, target_class]
                else:
                    output = output.max(dim=1)[0]
            else:
                output = outputs['predictions']
        else:
            output = outputs['predictions']
        
        # 反向传播
        output.backward(torch.ones_like(output))
        
        # 收集梯度
        gradients = self._collect_gradients(query_graph_with_grad)
        
        # 计算Saliency（取绝对值）
        saliency_maps = {k: v.abs() for k, v in gradients.items()}
        
        return saliency_maps

    def explain_integrated_gradients(self,
                                   query_graph,
                                   context_graphs: List,
                                   context_labels: torch.Tensor,
                                   task_type: str,
                                   baseline=None,
                                   steps: int = 50) -> Dict[str, torch.Tensor]:
        """
        使用Integrated Gradients方法
        
        baseline: 基线（默认为零）
        steps: 积分步数
        
        返回: 集成梯度字典
        """
        # 创建基线
        if baseline is None:
            baseline = self._create_baseline(query_graph)
        
        # 生成插值路径
        integrated_gradients = {}
        
        for step in range(steps):
            # 插值
            alpha = step / steps
            interpolated_graph = self._interpolate_graphs(
                baseline, query_graph, alpha
            )
            
            # 计算梯度
            gradients = self.explain_saliency(
                interpolated_graph,
                context_graphs,
                context_labels,
                task_type
            )
            
            # 累积梯度
            for key, grad in gradients.items():
                if key not in integrated_gradients:
                    integrated_gradients[key] = torch.zeros_like(grad)
                integrated_gradients[key] += grad / steps
        
        # 乘以输入与基线的差
        for key in integrated_gradients:
            input_features = self._get_features(query_graph, key)
            baseline_features = self._get_features(baseline, key)
            integrated_gradients[key] *= (input_features - baseline_features)
        
        return integrated_gradients

    def explain_smoothgrad(self,
                         query_graph,
                         context_graphs: List,
                         context_labels: torch.Tensor,
                         task_type: str,
                         noise_level: float = 0.1,
                         n_samples: int = 50) -> Dict[str, torch.Tensor]:
        """
        使用SmoothGrad方法减少噪声
        
        noise_level: 噪声水平
        n_samples: 采样次数
        
        返回: 平滑梯度字典
        """
        smooth_gradients = {}
        
        for _ in range(n_samples):
            # 添加噪声
            noisy_graph = self._add_noise(query_graph, noise_level)
            
            # 计算梯度
            gradients = self.explain_saliency(
                noisy_graph,
                context_graphs,
                context_labels,
                task_type
            )
            
            # 累积梯度
            for key, grad in gradients.items():
                if key not in smooth_gradients:
                    smooth_gradients[key] = torch.zeros_like(grad)
                smooth_gradients[key] += grad / n_samples
        
        return smooth_gradients

    def compute_cell_importance(self,
                              gradients: Dict[str, torch.Tensor],
                              aggregation: str = 'l2') -> Dict[str, torch.Tensor]:
        """
        计算单元格级别的重要性
        
        gradients: 梯度字典
        aggregation: 聚合方法 ('l1', 'l2', 'max')
        
        返回: 单元格重要性字典
        """
        cell_importance = {}
        
        for table_name, table_gradients in gradients.items():
            if aggregation == 'l1':
                # L1范数
                importance = table_gradients.abs().sum(dim=-1)
            elif aggregation == 'l2':
                # L2范数
                importance = torch.norm(table_gradients, p=2, dim=-1)
            elif aggregation == 'max':
                # 最大值
                importance = table_gradients.abs().max(dim=-1)[0]
            else:
                raise ValueError(f"未知的聚合方法: {aggregation}")
            
            cell_importance[table_name] = importance
        
        return cell_importance

    def visualize_node_importance(self,
                                gradients: Dict[str, torch.Tensor],
                                graph,
                                top_k: int = 10) -> Dict[str, Any]:
        """
        可视化节点重要性
        
        gradients: 梯度字典
        graph: 图结构
        top_k: 显示前k个重要节点
        
        返回: 可视化数据
        """
        # 计算节点重要性
        node_importance = []
        
        for node_id, node in graph.node_id_map.items():
            table = node.table
            if table in gradients:
                # 获取该节点的梯度
                node_grad = self._get_node_gradient(gradients[table], node_id)
                importance = node_grad.abs().mean().item()
                
                node_importance.append({
                    'node_id': node_id,
                    'table': table,
                    'importance': importance,
                    'features': node.features
                })
        
        # 排序并选择top-k
        node_importance.sort(key=lambda x: x['importance'], reverse=True)
        top_nodes = node_importance[:top_k]
        
        # 边重要性
        edge_importance = self._compute_edge_importance(gradients, graph)
        
        return {
            'node_importance': top_nodes,
            'edge_importance': edge_importance,
            'total_nodes': len(node_importance)
        }

    def _enable_gradients(self, graph):
        """为图中的特征启用梯度计算"""
        # 这里需要根据实际的图结构实现
        # 简化处理：假设图有features属性
        if hasattr(graph, 'node_features'):
            for table, features in graph.node_features.items():
                features.requires_grad_(True)
        return graph

    def _collect_gradients(self, graph) -> Dict[str, torch.Tensor]:
        """收集图中所有特征的梯度"""
        gradients = {}
        
        if hasattr(graph, 'node_features'):
            for table, features in graph.node_features.items():
                if features.grad is not None:
                    gradients[table] = features.grad.clone()
        
        return gradients

    def _create_baseline(self, graph):
        """创建基线图（零特征）"""
        # 创建与输入图相同结构但特征为零的图
        baseline = graph.clone() if hasattr(graph, 'clone') else graph
        
        if hasattr(baseline, 'node_features'):
            for table, features in baseline.node_features.items():
                baseline.node_features[table] = torch.zeros_like(features)
        
        return baseline

    def _interpolate_graphs(self, graph1, graph2, alpha: float):
        """在两个图之间插值"""
        interpolated = graph1.clone() if hasattr(graph1, 'clone') else graph1
        
        if hasattr(interpolated, 'node_features'):
            for table in interpolated.node_features:
                features1 = graph1.node_features[table]
                features2 = graph2.node_features[table]
                interpolated.node_features[table] = (
                    (1 - alpha) * features1 + alpha * features2
                )
        
        return interpolated

    def _add_noise(self, graph, noise_level: float):
        """向图添加高斯噪声"""
        noisy_graph = graph.clone() if hasattr(graph, 'clone') else graph
        
        if hasattr(noisy_graph, 'node_features'):
            for table, features in noisy_graph.node_features.items():
                noise = torch.randn_like(features) * noise_level
                noisy_graph.node_features[table] = features + noise
        
        return noisy_graph

    def _get_features(self, graph, key: str) -> torch.Tensor:
        """获取图中的特征"""
        if hasattr(graph, 'node_features') and key in graph.node_features:
            return graph.node_features[key]
        return torch.tensor(0.0)

    def _get_node_gradient(self, table_gradients: torch.Tensor, node_id: int) -> torch.Tensor:
        """获取特定节点的梯度"""
        # 假设node_id对应表中的行索引
        if node_id < table_gradients.shape[0]:
            return table_gradients[node_id]
        return torch.zeros(table_gradients.shape[-1])

    def _compute_edge_importance(self, gradients: Dict[str, torch.Tensor], graph) -> List[Dict[str, Any]]:
        """计算边的重要性"""
        edge_importance = []
        
        # 遍历所有边
        for edge_type, edges in graph.edges.items():
            for edge in edges:
                source_node = graph.node_id_map[edge.source]
                target_node = graph.node_id_map[edge.target]
                
                # 获取源和目标节点的梯度
                source_grad = self._get_node_gradient(
                    gradients.get(source_node.table, torch.tensor(0.0)),
                    edge.source
                )
                target_grad = self._get_node_gradient(
                    gradients.get(target_node.table, torch.tensor(0.0)),
                    edge.target
                )
                
                # 边重要性为两端节点重要性的平均
                importance = (source_grad.abs().mean() + target_grad.abs().mean()) / 2
                
                edge_importance.append({
                    'source': edge.source,
                    'target': edge.target,
                    'type': edge_type,
                    'importance': importance.item()
                })
        
        # 排序
        edge_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return edge_importance[:20]  # 返回前20条最重要的边


class ExplainabilityModule:
    """
    统一的解释性模块
    整合分析型和梯度型解释方法
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.analytical_explainer = AnalyticalExplainer()
        self.gradient_explainer = GradientExplainer(model)

    def explain(self,
               query_graph,
               context_graphs: List,
               context_labels: torch.Tensor,
               features: Dict[str, torch.Tensor],
               predictions: torch.Tensor,
               labels: torch.Tensor,
               task_type: str,
               column_metadata: Dict[str, Dict[str, Any]],
               method: str = 'combined') -> Dict[str, Any]:
        """
        生成解释
        
        method: 'analytical', 'gradient', 'combined'
        
        返回: 综合解释结果
        """
        explanation = {}
        
        if method in ['analytical', 'combined']:
            # 分析型解释
            global_explanation = self.analytical_explainer.explain_global(
                predictions, features, labels, column_metadata
            )
            explanation['analytical'] = global_explanation
        
        if method in ['gradient', 'combined']:
            # 梯度型解释
            saliency = self.gradient_explainer.explain_saliency(
                query_graph, context_graphs, context_labels, task_type
            )
            cell_importance = self.gradient_explainer.compute_cell_importance(saliency)
            
            explanation['gradient'] = {
                'saliency': saliency,
                'cell_importance': cell_importance
            }
        
        # 生成文本解释
        if 'analytical' in explanation:
            text_explanation = self.analytical_explainer.generate_text_explanation(
                explanation['analytical'],
                task_type=task_type
            )
            explanation['text'] = text_explanation
        
        return explanation
