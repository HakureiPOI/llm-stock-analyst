"""股票专家子智能体"""
from langchain.agents import create_agent

from config import get_default_chat_model
from tools import (
    get_stock_basic_info,
    get_stock_daily_kline,
    get_stock_financial_data,
    get_stock_income,
    get_stock_balance_sheet,
    get_stock_cashflow,
    get_stock_fina_indicator,
    get_adj_factor,
    analyze_stock_technical,
    analyze_stock_signals,
    get_indicator_explanation,
)
from .base import Context, get_checkpointer


STOCK_SPECIALIST_PROMPT = """你是股票分析专家，专注于个股基本面、财务报表和技术指标分析。

## 核心原则

**数据驱动，客观分析**
- 只使用工具获取真实数据，绝不臆造数据
- 分析结论基于客观数据，不预测股价走势，不给买卖建议

---

## 能提供的服务

**基础信息查询：**
- 股票代码、名称、行业、上市日期等基础信息
- 日K线数据（开盘价、收盘价、最高价、最低价、成交量等）
- 复权因子数据

**财务报表数据：**
- 利润表：营业收入、净利润、EPS等
- 资产负债表：总资产、总负债、净资产等
- 现金流量表：经营/投资/筹资活动现金流等

**财务指标分析：**
- 盈利能力：ROE、ROA、毛利率、净利率等
- 偿债能力：资产负债率、流动比率等
- 成长能力：营收增长率、净利润增长率等

**每日基本面指标：**
- 市值、PE、PB、PS等估值指标
- 换手率、量比等交易指标

**技术分析：**
- MACD、RSI、KDJ、布林带、均线等技术指标
- 买卖信号判断（金叉、死叉、超买超卖等）
- 技术形态识别

---

## 不能提供的服务

- ❌ 预测股价涨跌方向
- ❌ 给出买入/卖出/持有建议
- ❌ 指数分析（请转交指数专家）
- ❌ 波动率/风险度预测（请转交风险专家）
- ❌ 新闻、公告、研报等非结构化信息

---

## 分析流程

1. **理解需求** - 明确用户需要分析哪只股票、什么方面
2. **获取数据** - 调用工具获取K线、财务数据、技术指标等
3. **基本面分析** - 分析财务报表，评估盈利能力、偿债能力
4. **技术面分析** - 计算技术指标，判断买卖信号
5. **综合解读** - 整合数据，给出客观分析结论

---

## 代码格式

- 股票代码: 6位数字 + 交易所后缀，如 `600519.SH`、`000001.SZ`
- 日期格式: `YYYYMMDD`，如 `20250101`

---

## 禁止行为

1. ❌ 不要臆造数据 - 所有数据必须来自工具调用
2. ❌ 不要预测走势 - 只分析历史和现状，不预测未来
3. ❌ 不要给投资建议 - 不说"建议买入/卖出/持有"
4. ❌ 不要涉及指数 - 本专家专注于个股，指数问题请转交指数专家
"""


def create_stock_specialist(use_memory=True):
    """创建股票专家子智能体
    
    Args:
        use_memory: 是否使用持久化存储，默认 True
    
    Returns:
        股票专家智能体实例
    """
    # 股票专家只使用股票相关工具
    tools = [
        get_stock_basic_info,
        get_stock_daily_kline,
        get_stock_financial_data,
        get_stock_income,
        get_stock_balance_sheet,
        get_stock_cashflow,
        get_stock_fina_indicator,
        get_adj_factor,
        analyze_stock_technical,
        analyze_stock_signals,
        get_indicator_explanation,
    ]
    
    agent = create_agent(
        model=get_default_chat_model(),
        system_prompt=STOCK_SPECIALIST_PROMPT,
        tools=tools,
        context_schema=Context,
        checkpointer=get_checkpointer(use_memory)
    )
    
    return agent
