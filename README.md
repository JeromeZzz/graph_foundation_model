KumoRFM - 关系型基础模型

KumoRFM是一个针对关系型数据的基础模型实现，基于论文《KumoRFM: A Foundation Model for In-Context Learning on Relational Data》。该模型能够在任意关系型数据库上进行预测，无需针对特定任务进行训练。

主要特性

- 通用性: 支持任意关系型数据库结构和模式
- 多任务: 支持分类、回归、链接预测等多种预测任务
- 上下文学习: 通过少量示例即可进行准确预测
- 零代码: 使用PQL查询语言，无需编写代码
- 可解释性: 提供预测解释和特征重要性分析
- 高性能: 实时预测，秒级响应

项目结构

```
kumo_rfm/
├── config/              # 配置文件
├── data/               # 数据处理模块
├── models/             # 模型实现
│   ├── encoders/       # 编码器
│   ├── transformers/   # Transformer模块
│   └── icl/           # 上下文学习模块
├── pql/                # 预测查询语言
├── training/           # 训练脚本
├── evaluation/         # 评估模块
├── explainability/     # 可解释性模块
└── utils/              # 工具函数
```

安装

环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (可选，用于GPU加速)

安装步骤

创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

安装依赖
pip install -r requirements.txt

安装项目
pip install -e .
```

快速开始

1. 加载数据

```python
from kumo_rfm.data.database import RelationalDatabase
from kumo_rfm.utils.data_utils import load_csv_to_database

# 定义表和关系
csv_files = {
    'users': 'data/users.csv',
    'orders': 'data/orders.csv',
    'products': 'data/products.csv'
}

relationships = [
    ('orders', 'user_id', 'users', 'user_id'),
    ('orders', 'product_id', 'products', 'product_id')
]

# 加载数据库
database = load_csv_to_database(csv_files, relationships)
```

2. 初始化模型

```python
from kumo_rfm.config.model_config import KumoRFMConfig
from kumo_rfm.models.kumo_rfm import KumoRFM

# 创建配置
config = KumoRFMConfig()

# 初始化模型
model = KumoRFM(config)

# 使用数据库元数据初始化
model.initialize(
    column_metadata=database.get_column_metadata(),
    table_metadata=database.get_table_metadata(),
    num_node_types=len(database.tables),
    num_edge_types=len(database.relationships)
)
```

3. 执行预测

使用PQL (Predictive Query Language) 进行预测：

```python
from kumo_rfm.pql.parser import PQLExecutor

# 创建执行器
executor = PQLExecutor(database, model)

# 预测用户是否会在未来7天内下单
query = "PREDICT COUNT(orders.*, 0, 7) > 0 FOR users.user_id = 123"
result = executor.execute(query)

print(f"预测结果: {result}")
```

更多PQL示例：

```sql
-- 预测用户的生命周期价值
PREDICT SUM(orders.amount, 0, 365) FOR users.user_id IN (1, 2, 3)

-- 推荐产品
PREDICT LIST_DISTINCT(orders.product_id, 0, 7) FOR users.user_id = 456

-- 预测产品需求
PREDICT COUNT(orders.*, 0, 30) FOR products.product_id = 789
```

## 高级用法

### 微调模型

针对特定任务微调模型以获得更好的性能：

```python
from kumo_rfm.training.finetune import FineTuner

# 创建微调器
finetuner = FineTuner(
    config=config,
    model=model,
    database=database,
    pql_query="PREDICT COUNT(orders.*, 0, 7) > 0 FOR users.user_id"
)

# 执行微调
metrics = finetuner.finetune(num_epochs=10)
print(f"测试集性能: {metrics}")
```

### 模型解释

获取预测的解释：

```python
# 获取预测和解释
predictions = model.predict(
    database=database,
    entity_id=123,
    prediction_time=datetime.now(),
    task_config={'task_type': 'classification'}
)

explanations = model.explain(query_graph, predictions)
print(f"特征重要性: {explanations['feature_importance']}")
```

### 批量预测

对多个实体进行批量预测：

```python
# 批量预测
entities = [1, 2, 3, 4, 5]
batch_results = []

for entity_id in entities:
    result = executor.execute(
        f"PREDICT COUNT(orders.*, 0, 7) FOR users.user_id = {entity_id}"
    )
    batch_results.append(result)
```

## 模型架构

KumoRFM采用以下关键组件：

1. **多模态列编码器**: 支持数值、分类、文本、时间戳等多种数据类型
2. **表级Transformer**: 在列维度上应用注意力机制
3. **关系图Transformer**: 处理表之间的关系
4. **上下文学习模块**: 实现少样本学习能力

## 性能指标

在RELBENCH基准测试上的性能：

| 任务类型 | 平均AUROC | 平均MAE | 平均MAP@10 |
|---------|-----------|---------|------------|
| 分类    | 76.71     | -         | -             |
| 回归    | -            | 0.984  | -             |
| 推荐    | -            | -         | 7.29        |

## 配置选项

主要配置参数：

```python
config = KumoRFMConfig(
    hidden_dim=768,              # 隐藏层维度
    num_heads=12,                # 注意力头数
    num_layers=12,               # Transformer层数
    num_context_examples=32,     # 上下文示例数
    learning_rate=1e-4,          # 学习率
    batch_size=32,               # 批次大小
)
```

## 故障排除

### 常见问题

1. 内存不足
   - 减少`batch_size`
   - 减少`num_context_examples`
   - 使用梯度累积

2. 预测速度慢
   - 使用GPU加速
   - 减少图采样的跳数
   - 使用模型缓存

3. 精度不够
   - 增加上下文示例数
   - 对特定任务进行微调
   - 检查数据质量


## 联系方式

- 邮件: wzzhao@iaii.ac.cn