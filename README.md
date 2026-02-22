# LLM Stock Analyst

基于 LangGraph 的多智能体股票分析系统，集成了股票基本面分析、技术指标分析和波动率预测功能。

## 功能特性

- **多智能体架构**: Supervisor 协调股票专家、指数专家和波动率专家
- **股票分析**: 个股基本面、财务报表、技术指标分析
- **指数分析**: 大盘趋势、成分股权重、指数技术分析
- **波动率预测**: 基于 LightGBM 的波动率预测模型
- **联网搜索**: 集成阿里云百炼实时互联网搜索

## 项目结构

```
llm-stock-analyst/
├── agents/               # 智能体模块
│   ├── base.py          # 基础上下文和配置
│   ├── supervisor_entry.py  # 主智能体入口
│   ├── stock_specialist.py  # 股票专家
│   ├── index_specialist.py  # 指数专家
│   ├── volatility_specialist.py  # 波动率专家
│   ├── expert_tools.py   # 专家工具包装
│   └── agent_factory.py  # 智能体工厂
├── tools/                # LangChain 工具
│   ├── stock_tools.py    # 股票相关工具
│   ├── index_tools.py    # 指数相关工具
│   ├── volatility_tools.py  # 波动率预测工具
│   ├── analysis_tools.py  # 技术分析工具
│   ├── websearch_tools.py   # 联网搜索工具
│   └── common_tools.py   # 通用工具
├── tsdata/               # 数据源层
│   ├── stock.py          # 股票数据接口
│   ├── index.py          # 指数数据接口
│   ├── client.py         # Tushare 客户端
│   ├── indicators.py     # 技术指标计算
│   └── cache.py          # 数据缓存
├── ml/                   # 波动率预测模块
│   ├── feature_engineering.py  # 特征工程
│   ├── train_model.py    # 模型训练
│   ├── predict.py        # 模型预测
│   ├── get_stock_pool.py  # 股票池获取
│   └── get_stock_kline_dataset.py  # K线数据获取
└── config/               # 配置模块
    └── models.py         # 模型配置
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

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

# PostgreSQL URI (可选，用于持久化存储)
DB_URI=postgresql://user:password@host:port/database
```

### 3. 运行 LangGraph API

```bash
langgraph dev
```

或指定环境文件：

```bash
langgraph dev --env .env
```

## 使用示例

### 通过 API 调用

```python
from agents import create_supervisor_agent, Context

# 创建 Supervisor 智能体
supervisor = create_supervisor_agent(use_memory=False)

# 查询示例
response = supervisor.invoke(
    {"messages": [{"role": "user", "content": "分析贵州茅台的基本面"}]},
    context=Context(user_id="user_1")
)

print(response['messages'][-1].content)
```

### 预测股票波动率

```python
from ml.predict import VolatilityPredictor

# 创建预测器
predictor = VolatilityPredictor()

# 预测单只股票
result = predictor.predict(
    ts_code="600519.SH",
    days=365,
    window_size=5
)

print(f"R²: {result['metrics']['r2']:.4f}")
```

### 训练波动率模型

```bash
# 1. 获取股票池
python ml/get_stock_pool.py

# 2. 获取K线数据
python ml/get_stock_kline_dataset.py

# 3. 训练模型
python ml/train_model.py
```

## 智能体说明

### Supervisor (主智能体)
- 负责理解用户意图并协调专家智能体
- 根据问题类型分发给对应的专家

### Stock Specialist (股票专家)
- 个股基本面信息查询
- 财务报表分析（利润表、资产负债表、现金流量表）
- 技术指标分析（MACD、RSI、KDJ、布林带）
- 买卖信号判断

### Index Specialist (指数专家)
- 指数基础信息和趋势分析
- 指数成分股权重查询
- 大盘技术分析
- 板块轮动分析

### Volatility Specialist (波动率专家)
- 未来波动率预测
- 波动率模型表现分析
- 多股票波动率对比
- 波动率趋势分析

## 联网搜索工具

系统集成了基于阿里云通义千问的实时联网搜索能力，提供以下三种搜索工具：

### web_search (通用联网搜索)
- **功能**: 实时获取互联网信息，补充结构化数据的不足
- **适用场景**:
  - 获取最新的新闻资讯和突发事件
  - 查询实时市场动态和政策解读
  - 搜索公司最新公告和财报解读
  - 获取宏观经济数据和行业研究报告
  - 补充技术分析之外的实时信息
- **使用方式**: Supervisor 会根据问题自动调用，用户也可以在提问中明确要求"搜索相关信息"

### web_search_company (公司专属搜索)
- **功能**: 针对特定公司的定向搜索，自动构造优化查询
- **参数**:
  - `ts_code`: 股票代码 (如 600519.SH)
  - `query_type`: 搜索类型
    - `news`: 最新新闻 (默认)
    - `announcement`: 公告信息
    - `analysis`: 分析报告
    - `general`: 综合信息
- **优势**: 自动提取股票代码并构造专业搜索关键词，返回更精准的结果

### web_search_market (大盘搜索)
- **功能**: 搜索大盘指数的最新市场动态和趋势分析
- **参数**:
  - `index_code`: 指数代码，默认上证指数 (000001.SH)
  - 支持的指数: 上证指数、深证成指、创业板指、沪深300、上证50、中证500
- **优势**: 针对指数分析优化，自动匹配指数名称并构造搜索查询

### 数据源优先级
Supervisor 在分析时会遵循以下数据源优先级：
1. **结构化数据 (tsdata)**: Tushare 提供的历史数据、财务报表、技术指标等，作为分析基础
2. **联网搜索 (websearch)**: 作为辅助工具，用于获取最新资讯、市场动态和实时信息
3. **交叉验证**: 将结构化数据与联网搜索结果进行对比，发现数据不一致点并标注

### 使用示例

```python
# 示例1: 分析贵州茅台，Supervisor会自动组合使用数据源和联网搜索
response = supervisor.invoke(
    {"messages": [{"role": "user", "content": "分析贵州茅台的最新基本面和近期新闻"}]},
    context=Context(user_id="user_1")
)

# 示例2: 查询大盘趋势，Supervisor会自动搜索最新市场动态
response = supervisor.invoke(
    {"messages": [{"role": "user", "content": "上证指数最近有什么重要新闻"}]},
    context=Context(user_id="user_1")
)

# 示例3: 用户明确要求使用联网搜索
response = supervisor.invoke(
    {"messages": [{"role": "user", "content": "搜索宁德时代最近的分析报告"}]},
    context=Context(user_id="user_1")
)
```

## API 工具

| 工具名称 | 功能 | 所属专家 |
|---------|------|---------|
| `get_stock_basic_info` | 查询股票基础信息 | Stock Specialist |
| `get_stock_daily_kline` | 获取股票日K线 | Stock Specialist |
| `get_stock_financial_data` | 获取每日基本面数据 | Stock Specialist |
| `get_stock_income` | 获取利润表 | Stock Specialist |
| `analyze_stock_technical` | 技术指标分析 | Stock Specialist |
| `get_index_basic_info` | 查询指数基础信息 | Index Specialist |
| `get_index_weight` | 获取指数成分股权重 | Index Specialist |
| `predict_stock_volatility` | 预测股票波动率 | Volatility Specialist |
| `compare_stock_volatility` | 对比多只股票波动率 | Volatility Specialist |
| `web_search` | 通用联网搜索 | Supervisor |
| `web_search_company` | 搜索公司相关信息 | Supervisor |
| `web_search_market` | 搜索大盘市场动态 | Supervisor |

## 数据源

- **Tushare**: 提供股票、指数、财务数据等结构化数据
- **DashScope**: 通义千问大语言模型 + 联网搜索能力

## 技术栈

- **LangGraph**: 多智能体编排框架
- **LangChain**: LLM 应用开发框架
- **DashScope**: 阿里云通义千问大模型 + 联网搜索
- **LightGBM**: 波动率预测模型
- **Pandas/NumPy**: 数据处理
- **Tushare**: A股数据接口

## 注意事项

1. 股票代码格式: 6位数字 + 交易所(SH/SZ)，如 `600519.SH`
2. 日期格式: YYYYMMDD，如 `20250101`
3. 投资有风险，所有分析仅供参考，不构成投资建议


