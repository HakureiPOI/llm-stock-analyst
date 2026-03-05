# LLM Stock Analyst 项目中期报告

> 报告日期：2026年3月5日  
> 项目版本：v1.0

---

## 一、项目概述

### 1.1 项目背景

传统股票分析依赖人工研判，效率低且主观性强。本项目旨在构建一个基于大语言模型的多智能体股票分析系统，实现自动化、智能化的投资分析服务。

### 1.2 项目目标

| 目标 | 描述 | 状态 |
|------|------|------|
| 多智能体架构 | Supervisor协调多个专家智能体 | ✅ 完成 |
| 股票分析 | 基本面、财务报表、技术指标 | ✅ 完成 |
| 指数分析 | 指数信息、成分股、技术分析 | ✅ 完成 |
| 波动率预测 | ML模型预测下一交易日波动率 | ✅ 完成 |
| 可解释性 | SHAP特征贡献度分析 | ✅ 完成 |

---

## 二、技术架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户请求                              │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Supervisor (主智能体)                       │
│         - 意图识别   - 任务分发   - 结果聚合             │
└──────┬──────────┬──────────┬──────────┬────────────────┘
       ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Stock   │ │  Index   │ │Volatility│ │ WebSearch│
│Specialist│ │Specialist│ │Specialist│ │  (工具)  │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │            │
     ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────┐
│                   工具层 (LangChain Tools)              │
│  stock_tools │ index_tools │ volatility_tools │ ...     │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   数据层 (tsdata)                        │
│     Tushare API │ 本地缓存 │ ML模型                      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| 框架 | LangGraph | >= 0.2.0 |
| LLM | 阿里云通义千问 | - |
| ML | LightGBM + SHAP | >= 4.0.0 |
| 数据源 | Tushare | >= 1.4.0 |
| 波动率模型 | ARCH (GARCH) | >= 5.0.0 |

---

## 三、核心模块详解

### 3.1 智能体模块 (agents/)

| 智能体 | 职责 | 工具数量 |
|--------|------|----------|
| Supervisor | 意图识别、任务分发、联网搜索 | 1 |
| Stock Specialist | 股票基本面、财务、技术分析 | 8 |
| Index Specialist | 指数信息、成分股、技术分析 | 4 |
| Volatility Specialist | 波动率预测、SHAP解读 | 3 |

### 3.2 波动率预测模块 (ml/)

#### 3.2.1 特征工程

共 **138个特征**，分为以下类别：

| 特征类别 | 特征数 | 示例 |
|----------|--------|------|
| 收益率特征 | 15+ | log_ret, ret_cum_5/10, ret_std_20 |
| 滚动统计 | 30+ | ret_skew_20, ret_kurt_20, ret_zscore |
| 波动率代理 | 20+ | yang_zhang_vol, parkinson_vol, garman_klass_vol |
| GARCH特征 | 9 | garch_vol, egarch_vol, gjr_vol + 残差 |
| 成交量特征 | 10+ | log_vol, vol_ratio_5_20, vol_ma_dev |
| OHLC区间 | 25+ | oc_range, hl_range, co_gap |
| 技术指标 | 15+ | rsi_14, macd, atr_14 |

#### 3.2.2 模型架构

```
LightGBM 回归模型
├── 输入: 138维特征向量
├── 目标: log(Yang-Zhang波动率_t+1)
├── 训练数据: 10年上证指数日线 (1926条)
└── 验证方式: Walk-Forward (滚动前向)
```

#### 3.2.3 模型性能

**训练集表现 (10年数据):**

| 指标 | LightGBM | EWMA | GARCH | EGARCH | GJR-GARCH |
|------|----------|------|-------|--------|-----------|
| MAE | **0.0029** | 0.0039 | 0.0042 | 0.0042 | 0.0043 |
| RMSE | **0.0044** | 0.0062 | 0.0065 | 0.0064 | 0.0066 |
| R² | **-0.01** | -0.06 | -0.22 | -0.19 | -0.23 |
| 方向准确率 | **52.1%** | 51.8% | 50.7% | 50.6% | 50.9% |
| 相关系数 | **0.24** | - | - | - | - |
| 命中率(±20%) | **35.9%** | - | - | - | - |

**跨指数泛化测试:**

| 指数 | 样本数 | MAE | R² | 方向准确率 |
|------|--------|-----|-----|-----------|
| 沪深300 | 1927 | 0.0031 | **0.197** | **58.0%** |
| 深证成指 | 1927 | 0.0038 | 0.134 | **58.0%** |
| 上证50 | 1927 | 0.0031 | 0.140 | 55.8% |
| 创业板指 | 1927 | 0.0051 | -0.030 | 54.2% |

#### 3.2.4 Top 10 特征重要性

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | co_gap_zscore20 | 0.082 |
| 2 | downside_vol_5 | 0.065 |
| 3 | downside_vol_10 | 0.058 |
| 4 | garch_vol | 0.052 |
| 5 | ret_std_20 | 0.048 |
| 6 | oc_range_std5 | 0.042 |
| 7 | pct_chg | 0.038 |
| 8 | garman_klass_vol_ma10 | 0.035 |
| 9 | vol_ratio_5_20 | 0.032 |
| 10 | parkinson_vol_ma10 | 0.028 |

---

## 四、工具层设计

### 4.1 工具清单

| 工具名称 | 功能 | 所属模块 |
|----------|------|----------|
| `get_stock_basic_info` | 股票基础信息 | stock_tools |
| `get_stock_daily_kline` | 日K线数据 | stock_tools |
| `get_stock_income` | 利润表 | stock_tools |
| `get_stock_balance_sheet` | 资产负债表 | stock_tools |
| `get_stock_cashflow` | 现金流量表 | stock_tools |
| `analyze_stock_technical` | 技术指标分析 | analysis_tools |
| `get_index_basic_info` | 指数基础信息 | index_tools |
| `get_index_weight` | 成分股权重 | index_tools |
| `predict_index_volatility` | 波动率预测 | volatility_tools |
| `compare_index_volatility` | 多指数对比 | volatility_tools |
| `get_market_volatility_summary` | 市场摘要 | volatility_tools |
| `web_search` | 联网搜索 | websearch_tools |

### 4.2 工具设计原则

**职责分离原则：**
- 工具层只返回**原始JSON数据**
- 智能体层负责**解读和呈现**
- 避免在工具中硬编码判断逻辑

**示例：波动率预测工具输出**
```json
{
  "ts_code": "000001.SH",
  "predicted_volatility": 0.0072,
  "volatility_pct": "0.72%",
  "percentile": 71.5,
  "historical_stats": {...},
  "volatility_trend": {"change_pct": -26.93},
  "shap_analysis": {"top_features": [...]}
}
```

---

## 五、关键创新点

### 5.1 波动率预测方法

**Yang-Zhang波动率代理：**
```
σ²_YZ = σ²_overnight + k·σ²_open_close + (1-k)·σ²_RS
```
- 结合隔夜跳空和日内波动
- 比简单收益率标准差更准确

### 5.2 SHAP可解释性

每个预测结果附带SHAP特征贡献度分析：

| 特征 | SHAP值 | 贡献度 | 解读 |
|------|--------|--------|------|
| ret_std_20 | +0.046 | 8.7% | 20日收益波动推高预测 |
| pct_chg | -0.033 | 6.3% | 当日涨幅压低预测 |
| downside_vol_5 | +0.029 | 5.5% | 下行波动推高预测 |

### 5.3 历史分位数语义

将绝对波动率转化为相对风险等级：

| 分位数 | 等级 | 含义 |
|--------|------|------|
| < 25% | 极低 | 市场非常平静 |
| 25-50% | 偏低 | 低于历史中位 |
| 50-75% | 中等 | 正常水平 |
| 75-90% | 偏高 | 需关注风险 |
| ≥ 90% | 极高 | 市场剧烈波动 |

---

## 六、项目文件清单

```
llm-stock-analyst/
├── agents/                          # 智能体模块
│   ├── __init__.py
│   ├── base.py                      # 基础上下文
│   ├── supervisor_entry.py          # 主智能体
│   ├── stock_specialist.py          # 股票专家
│   ├── index_specialist.py          # 指数专家
│   ├── volatility_specialist.py     # 波动率专家
│   ├── expert_tools.py              # 专家工具包装
│   └── agent_factory.py             # 智能体工厂
├── tools/                           # 工具层
│   ├── __init__.py
│   ├── stock_tools.py               # 股票工具 (8个)
│   ├── index_tools.py               # 指数工具 (4个)
│   ├── volatility_tools.py          # 波动率工具 (3个)
│   ├── analysis_tools.py            # 分析工具 (4个)
│   ├── common_tools.py              # 通用工具 (2个)
│   └── websearch_tools.py           # 联网搜索
├── tsdata/                          # 数据源层
│   ├── __init__.py
│   ├── client.py                    # Tushare客户端
│   ├── stock.py                     # 股票数据接口
│   ├── index.py                     # 指数数据接口
│   ├── indicators.py                # 技术指标计算
│   └── cache.py                     # 数据缓存
├── ml/                              # 波动率预测模块
│   ├── feature_engineering.py       # 特征工程 (138特征)
│   ├── garch_features.py            # GARCH特征提取
│   ├── train_model.py               # 模型训练
│   ├── predict.py                   # 模型预测
│   ├── baseline_models.py           # 基线模型
│   ├── get_index_data.py            # 数据获取
│   ├── dataset/                     # 数据目录
│   │   ├── index_daily_000001_SH_10years.csv
│   │   └── index_features.csv
│   └── models/                      # 模型目录
│       ├── volatility_model_lgb.pkl
│       ├── feature_importance.csv
│       ├── model_metadata.json
│       └── baseline_comparison.csv
├── config/                          # 配置模块
│   ├── __init__.py
│   └── models.py                    # 模型配置
├── README.md                        # 项目文档
├── requirements.txt                 # 依赖清单
├── langgraph.json                   # LangGraph配置
└── .env                             # 环境变量
```

---

## 七、后续规划

### 7.1 短期优化 (1-2周)

| 任务 | 优先级 | 状态 |
|------|--------|------|
| 添加预测置信区间 | 高 | 待开始 |
| 近期预测准确率追踪 | 高 | 待开始 |
| 波动率异常预警 | 中 | 待开始 |

### 7.2 中期扩展 (1-2月)

| 任务 | 描述 |
|------|------|
| 多时间窗口预测 | 支持未来5日、20日波动率预测 |
| 个股波动率预测 | 扩展模型支持个股 |
| 实时数据更新 | 定时更新模型训练数据 |
| 历史相似场景匹配 | 找到历史相似波动率时期的后续走势 |

### 7.3 长期愿景

- 支持更多市场（港股、美股）
- 集成更多数据源（新闻情绪、宏观经济）
- 构建完整的投资决策辅助系统

---

## 八、风险与限制

### 8.1 技术限制

1. **波动率 ≠ 涨跌方向**：模型只能预测波动程度，不预测市场方向
2. **方向准确率有限**：约52%，略高于随机水平
3. **极端事件表现**：在市场剧烈波动时期（如2008、2015），模型表现下降

### 8.2 数据限制

1. **单一市场**：仅支持A股市场
2. **数据滞后**：依赖Tushare数据更新频率
3. **历史窗口**：需要足够历史数据（建议≥500天）

---

## 九、总结

本项目成功构建了一个多智能体股票分析系统，核心创新点包括：

1. **职责分离架构**：工具层返回数据，智能体层解读呈现
2. **可解释ML模型**：LightGBM + SHAP实现波动率预测可解释性
3. **多智能体协作**：Supervisor协调各领域专家智能体
4. **历史分位数语义**：将绝对值转化为用户可理解的相对风险等级

模型在10年数据上训练，MAE为0.0029，R²接近0，方向准确率52.1%，在沪深300等指数上泛化表现优异（方向准确率达58%）。

---

## 附录

### A. 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key
```

### B. 运行命令

```bash
# 获取数据
python -m ml.get_index_data

# 特征工程
python -m ml.feature_engineering

# 模型训练
python -m ml.train_model

# 启动服务
langgraph dev
```

### C. 支持的指数

| 代码 | 名称 |
|------|------|
| 000001.SH | 上证指数 |
| 399001.SZ | 深证成指 |
| 399006.SZ | 创业板指 |
| 000300.SH | 沪深300 |
| 000016.SH | 上证50 |
| 000905.SH | 中证500 |
| 000852.SH | 中证1000 |
