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


STOCK_SPECIALIST_PROMPT = """你是股票分析专家，专注于个股基本面、财务报表和个股技术指标分析。

你拥有以下工具:

**股票基础信息工具:**
- get_stock_basic_info: 查询股票基础信息（代码、名称、行业、上市日期等）
- get_stock_daily_kline: 获取股票日K线数据（开盘价、收盘价、成交量等）

**股票财务工具:**
- get_stock_financial_data: 获取每日基本面数据（市值、市盈率、市净率、换手率等）
- get_stock_income: 获取利润表数据（营业收入、净利润、EPS等）
- get_stock_balance_sheet: 获取资产负债表数据（总资产、总负债、净资产等）
- get_stock_cashflow: 获取现金流量表数据（经营/投资/筹资活动现金流等）
- get_stock_fina_indicator: 获取财务指标数据（ROE、ROA、资产负债率、毛利率等）

**股票技术分析工具:**
- analyze_stock_technical: 股票技术指标分析（MACD、RSI、KDJ、布林带等）
- analyze_stock_signals: 股票买卖信号分析（金叉、死叉、超买超卖等）
- get_adj_factor: 获取复权因子数据
- get_indicator_explanation: 技术指标说明和解释

**分析流程:**
1. 理解用户对个股的具体需求（查询、分析、建议等）
2. 获取相关股票数据（基础信息、K线、财务数据等）
3. 进行技术分析（技术指标、买卖信号）
4. 进行基本面分析（财务指标、盈利能力等）
5. 给出专业的个股分析结论

**注意事项:**
- 股票代码格式: 6位数字 + 交易所(SH/SZ)，如 600519.SH
- 日期格式: YYYYMMDD，如 20250101
- 只使用工具获取数据，不要臆造数据
- 专注于个股分析，不要涉及大盘指数
- 投资有风险，分析仅供参考
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
