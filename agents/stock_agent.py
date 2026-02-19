"""股票分析智能体"""
import os
import sys
from pathlib import Path
import dotenv
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.postgres import PostgresSaver

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import (
    # 股票工具
    get_stock_basic_info,
    get_stock_daily_kline,
    get_stock_financial_data,
    get_stock_income,
    get_stock_balance_sheet,
    get_stock_cashflow,
    get_stock_fina_indicator,
    get_adj_factor,
    get_cache_status,
    # 指数工具
    get_index_basic_info,
    get_index_daily_kline,
    get_index_weight,
    get_index_dailybasic,
    # 分析工具
    analyze_stock_technical,
    analyze_index_technical,
    analyze_stock_signals,
    get_indicator_explanation,
    # 通用工具
    get_current_time,
    get_stock_market_status,
)

dotenv.load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DB_URI = os.getenv("DB_URI")

# 系统提示词
SYSTEM_PROMPT = """你是一位专业的股票分析师和投资顾问，擅长技术分析、基本面分析和市场趋势判断。

你拥有以下工具:

**通用工具:**
- get_current_time: 获取当前世界时间，包括UTC时间和本地时间
- get_stock_market_status: 获取当前中国股市交易状态（开市/休市）

**股票数据工具:**
- get_stock_basic_info: 查询股票基础信息（代码、名称、行业、上市日期等）
- get_stock_daily_kline: 获取股票日K线数据（开盘价、收盘价、成交量等）
- get_stock_financial_data: 获取每日基本面数据（市值、市盈率、市净率、换手率等）
- get_stock_income: 获取利润表数据（营业收入、净利润、EPS等）
- get_stock_balance_sheet: 获取资产负债表数据（总资产、总负债、净资产等）
- get_stock_cashflow: 获取现金流量表数据（经营/投资/筹资活动现金流等）
- get_stock_fina_indicator: 获取财务指标数据（ROE、ROA、资产负债率、毛利率等）
- get_adj_factor: 获取复权因子数据

**指数数据工具:**
- get_index_basic_info: 查询指数基础信息
- get_index_daily_kline: 获取指数日K线数据
- get_index_weight: 获取指数成分股权重
- get_index_dailybasic: 获取指数每日基本面数据

**技术分析工具:**
- analyze_stock_technical: 股票技术指标分析（MACD、RSI、KDJ、布林带等）
- analyze_index_technical: 指数技术指标分析
- analyze_stock_signals: 股票买卖信号分析（金叉、死叉、超买超卖等）
- get_indicator_explanation: 技术指标说明和解释

**分析流程:**
1. 理解用户需求（查询、分析、建议等）
2. 获取相关数据（股票/指数基础信息、K线、财务数据等）
3. 进行技术分析（技术指标、买卖信号）
4. 进行基本面分析（财务指标、盈利能力等）
5. 综合分析给出专业建议

**注意事项:**
- 股票代码格式: 6位数字 + 交易所(SH/SZ)，如 600519.SH
- 日期格式: YYYYMMDD，如 20250101
- 只使用工具获取数据，不要臆造数据
- 投资有风险，分析仅供参考
"""

# 上下文定义
@dataclass
class Context:
    """自定义运行时上下文"""
    user_id: str = '1'
    session_id: Optional[str] = None


# 响应格式定义（暂时禁用结构化输出，改用普通文本）
# @dataclass
# class AnalysisResponse:
#     """分析响应格式"""
#     summary: str
#     technical_analysis: Optional[str] = None
#     fundamental_analysis: Optional[str] = None
#     recommendation: Optional[str] = None
#     risk_warning: Optional[str] = None


# 创建智能体
def create_stock_agent(use_memory=True):
    """创建股票分析智能体"""
    
    # 配置模型
    model = init_chat_model(
        model="qwen-turbo",
        model_provider="openai",
        temperature=0.7,
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 所有工具
    tools = [
        # 股票工具
        get_stock_basic_info,
        get_stock_daily_kline,
        get_stock_financial_data,
        get_stock_income,
        get_stock_balance_sheet,
        get_stock_cashflow,
        get_stock_fina_indicator,
        get_adj_factor,
        get_cache_status,
        # 指数工具
        get_index_basic_info,
        get_index_daily_kline,
        get_index_weight,
        get_index_dailybasic,
        # 分析工具
        analyze_stock_technical,
        analyze_index_technical,
        analyze_stock_signals,
        get_indicator_explanation,
        # 通用工具
        get_current_time,
        get_stock_market_status,
    ]
    
    # 选择检查点保存器
    if use_memory and DB_URI:
        checkpointer = PostgresSaver.from_conn_string(DB_URI)
        checkpointer.setup()
    else:
        checkpointer = InMemorySaver()
    
    # 创建智能体
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        context_schema=Context,
        # response_format=ToolStrategy(AnalysisResponse),  # 暂时禁用结构化输出
        # checkpointer=checkpointer     # LangSmith测试暂时禁用checkpoint
    )
    
    return agent

agent = create_stock_agent(use_memory=False)

