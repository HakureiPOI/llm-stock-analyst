"""指数专家子智能体"""
from langchain.agents import create_agent

from config import get_default_chat_model
from tools import (
    get_index_basic_info,
    get_index_daily_kline,
    get_index_weight,
    get_index_dailybasic,
    analyze_index_technical,
    get_indicator_explanation,
)
from .base import Context, get_checkpointer


INDEX_SPECIALIST_PROMPT = """你是指数分析专家，专注于大盘趋势、指数权重及成分股分析。

你拥有以下工具:

**指数基础信息工具:**
- get_index_basic_info: 查询指数基础信息
- get_index_daily_kline: 获取指数日K线数据

**指数成分分析工具:**
- get_index_weight: 获取指数成分股权重
- get_index_dailybasic: 获取指数每日基本面数据

**指数技术分析工具:**
- analyze_index_technical: 指数技术指标分析（MACD、RSI、KDJ、布林带等）
- get_indicator_explanation: 技术指标说明和解释

**分析流程:**
1. 理解用户对指数的具体需求（查询、分析、建议等）
2. 获取相关指数数据（基础信息、K线、成分股等）
3. 进行技术分析（技术指标、趋势判断）
4. 分析成分股权重和板块轮动
5. 给出专业的指数分析结论

**注意事项:**
- 指数代码格式: 6位数字 + 交易所(SH/SZ)，如 000001.SH
- 日期格式: YYYYMMDD，如 20250101
- 只使用工具获取数据，不要臆造数据
- 专注于指数和大盘分析，不要涉及个股
- 投资有风险，分析仅供参考
"""


def create_index_specialist(use_memory=True):
    """创建指数专家子智能体
    
    Args:
        use_memory: 是否使用持久化存储，默认 True
    
    Returns:
        指数专家智能体实例
    """
    # 指数专家只使用指数相关工具
    tools = [
        get_index_basic_info,
        get_index_daily_kline,
        get_index_weight,
        get_index_dailybasic,
        analyze_index_technical,
        get_indicator_explanation,
    ]
    
    agent = create_agent(
        model=get_default_chat_model(),
        system_prompt=INDEX_SPECIALIST_PROMPT,
        tools=tools,
        context_schema=Context,
        checkpointer=get_checkpointer(use_memory)
    )
    
    return agent
