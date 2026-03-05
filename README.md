# LLM Stock Analyst

基于 LangGraph 的多智能体股票分析系统，集成了股票基本面分析、技术指标分析和指数波动率预测功能。

## 功能特性

- **多智能体架构**: Supervisor 协调股票专家、指数专家和波动率专家
- **股票分析**: 个股基本面、财务报表、技术指标分析
- **指数分析**: 大盘趋势、成分股权重、指数技术分析
- **波动率预测**: 基于 LightGBM + GARCH 的下一交易日波动率预测
- **SHAP可解释性**: 提供特征贡献度分析，解释预测原因
- **联网搜索**: 集成阿里云百炼实时互联网搜索

## 项目结构

```
llm-stock-analyst/
├── agents/                    # 智能体模块
│   ├── base.py               # 基础上下文和配置
│   ├── supervisor_entry.py   # 主智能体入口
│   ├── stock_specialist.py   # 股票专家
│   ├── index_specialist.py   # 指数专家
│   └── volatility_specialist.py  # 波动率专家
├── tools/                     # LangChain 工具
│   ├── stock_tools.py        # 股票相关工具
│   ├── index_tools.py        # 指数相关工具
│   ├── volatility_tools.py   # 波动率预测工具
│   ├── analysis_tools.py     # 技术分析工具
│   ├── websearch_tools.py    # 联网搜索工具
│   └── common_tools.py       # 通用工具
├── tsdata/                    # 数据源层
│   ├── stock.py              # 股票数据接口
│   ├── index.py              # 指数数据接口
│   ├── indicators.py         # 技术指标计算
│   └── cache.py              # 数据缓存
├── ml/                        # 波动率预测模块
│   ├── feature_engineering.py    # 特征工程 (138个特征)
│   ├── garch_features.py     # GARCH波动率特征
│   ├── train_model.py        # 模型训练 (Walk-Forward)
│   ├── predict.py            # 模型预测
│   └── baseline_models.py    # 基线模型对比
└── config/                    # 配置模块
    └── models.py             # 模型配置
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# DashScope API Key (通义千问)
DASHSCOPE_API_KEY=your_dashscope_api_key

# Tushare API Token
TUSHARE_TOKEN=your_tushare_token
```

### 3. 训练模型（首次使用）

```bash
# 获取指数数据（默认10年）
python -m ml.get_index_data

# 特征工程
python -m ml.feature_engineering

# 模型训练
python -m ml.train_model
```

### 4. 运行 LangGraph API

```bash
langgraph dev
```

## 智能体说明

| 智能体 | 职责 |
|-------|------|
| **Supervisor** | 理解用户意图，协调专家智能体 |
| **Stock Specialist** | 个股基本面、财务报表、技术指标 |
| **Index Specialist** | 指数信息、成分股权重、大盘分析 |
| **Volatility Specialist** | 波动率预测、风险评估、SHAP解读 |

## 波动率预测

### 预测目标
- **下一交易日 Yang-Zhang 波动率**
- 结合隔夜跳空和日内波动，比简单标准差更准确

### 模型架构
- **LightGBM** + **GARCH特征**
- 138个特征（滚动统计、波动率代理、技术指标等）
- Walk-Forward 验证，10年数据训练

### 模型性能（10年数据回测）

| 指标 | LightGBM | EWMA | GARCH |
|------|----------|------|-------|
| MAE | **0.0029** | 0.0039 | 0.0042 |
| R² | **-0.01** | -0.06 | -0.19 |
| 方向准确率 | 52.1% | 51.8% | 50.6% |

### SHAP特征分析
预测结果包含SHAP特征贡献度，解释预测原因：
- `ret_std_20`: 20日收益波动
- `downside_vol`: 下行波动
- `co_gap`: 隔夜跳空
- `garch_vol`: GARCH估计波动

### 历史分位数语义

| 分位数 | 风险等级 | 含义 |
|--------|---------|------|
| < 25% | 极低 | 市场非常平静 |
| 25-50% | 偏低 | 低于历史中位 |
| 50-75% | 中等 | 正常水平 |
| 75-90% | 偏高 | 需关注风险 |
| >= 90% | 极高 | 市场剧烈波动 |

## API 工具

| 工具 | 功能 |
|------|------|
| `predict_index_volatility` | 预测单指数波动率 |
| `compare_index_volatility` | 对比多指数波动率 |
| `get_market_volatility_summary` | 市场波动率摘要 |
| `get_stock_basic_info` | 股票基础信息 |
| `get_stock_daily_kline` | 股票日K线 |
| `get_index_basic_info` | 指数基础信息 |
| `analyze_stock_technical` | 技术指标分析 |
| `web_search` | 联网搜索 |

## 使用示例

### Python 调用

```python
from agents import create_supervisor_agent, Context

supervisor = create_supervisor_agent(use_memory=False)

# 波动率预测
response = supervisor.invoke(
    {"messages": [{"role": "user", "content": "预测上证指数明天的波动率"}]},
    context=Context(user_id="user_1")
)

# 股票分析
response = supervisor.invoke(
    {"messages": [{"role": "user", "content": "分析贵州茅台的基本面"}]},
    context=Context(user_id="user_1")
)
```

### ML 模块直接使用

```python
from ml.predict import VolatilityPredictor

predictor = VolatilityPredictor()
result = predictor.predict("000001.SH", days=300, include_shap=True)

print(f"预测波动率: {result['volatility_pct']}")
print(f"历史分位: {result['percentile']}%")
```

## 支持的指数

| 代码 | 名称 |
|------|------|
| 000001.SH | 上证指数 |
| 399001.SZ | 深证成指 |
| 399006.SZ | 创业板指 |
| 000300.SH | 沪深300 |
| 000016.SH | 上证50 |
| 000905.SH | 中证500 |
| 000852.SH | 中证1000 |

## 技术栈

- **LangGraph**: 多智能体编排框架
- **LangChain**: LLM 应用开发
- **LightGBM**: 波动率预测模型
- **SHAP**: 模型可解释性
- **ARCH**: GARCH波动率模型
- **Tushare**: A股数据接口
- **DashScope**: 通义千问大模型

## 注意事项

1. 股票代码格式: 6位数字 + 交易所(SH/SZ)，如 `600519.SH`
2. **波动率 ≠ 涨跌方向**：波动率仅反映波动程度
3. 投资有风险，所有分析仅供参考
