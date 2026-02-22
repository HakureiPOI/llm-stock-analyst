"""专家智能体工具包装 - 将子智能体包装为工具供Supervisor调用"""
from langchain.tools import tool

from .stock_specialist import create_stock_specialist
from .index_specialist import create_index_specialist
from .volatility_specialist import create_volatility_specialist
from .base import Context


# 创建子智能体实例（使用内存存储）
_stock_specialist = create_stock_specialist(use_memory=False)
_index_specialist = create_index_specialist(use_memory=False)
_volatility_specialist = create_volatility_specialist(use_memory=False)


@tool
def call_stock_expert(query: str) -> str:
    """当需要查询个股基本面、财务报表或个股技术分析时，调用此工具。

    适用于以下场景:
    - 查询某只股票的基础信息
    - 分析个股的K线数据
    - 查询财务报表（利润表、资产负债表、现金流量表）
    - 个股技术指标分析（MACD、RSI、KDJ等）
    - 个股买卖信号判断

    Args:
        query: 用户对个股的具体问题或需求

    Returns:
        股票专家的分析结果
    """
    try:
        response = _stock_specialist.invoke(
            {"messages": [{"role": "user", "content": query}]},
            context=Context(user_id="stock_expert")
        )
        # 返回最后一条消息的内容
        messages = response.get('messages', [])
        if messages:
            last_msg = messages[-1]
            return last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        return "股票专家返回了空结果"
    except Exception as e:
        return f"股票专家调用失败: {str(e)}"


@tool
def call_index_expert(query: str) -> str:
    """当需要查询大盘指数、行业指数或指数成分股权重时，调用此工具。

    适用于以下场景:
    - 查询指数基础信息
    - 分析指数K线数据和趋势
    - 查询指数成分股权重
    - 指数技术指标分析
    - 大盘趋势判断

    Args:
        query: 用户对指数的具体问题或需求

    Returns:
        指数专家的分析结果
    """
    try:
        response = _index_specialist.invoke(
            {"messages": [{"role": "user", "content": query}]},
            context=Context(user_id="index_expert")
        )
        # 返回最后一条消息的内容
        messages = response.get('messages', [])
        if messages:
            last_msg = messages[-1]
            return last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        return "指数专家返回了空结果"
    except Exception as e:
        return f"指数专家调用失败: {str(e)}"


@tool
def call_volatility_expert(query: str) -> str:
    """当需要预测或分析股票波动率时，调用此工具。

    适用于以下场景:
    - 预测某只股票的未来波动率
    - 分析波动率预测模型的表现
    - 对比多只股票的波动率特征
    - 查询股票波动率趋势

    Args:
        query: 用户对波动率的具体问题或需求，应包含股票代码

    Returns:
        波动率专家的分析结果
    """
    try:
        response = _volatility_specialist.invoke(
            {"messages": [{"role": "user", "content": query}]},
            context=Context(user_id="volatility_expert")
        )
        # 返回最后一条消息的内容
        messages = response.get('messages', [])
        if messages:
            last_msg = messages[-1]
            return last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        return "波动率专家返回了空结果"
    except Exception as e:
        return f"波动率专家调用失败: {str(e)}"


__all__ = [
    "call_stock_expert",
    "call_index_expert",
    "call_volatility_expert",
]

