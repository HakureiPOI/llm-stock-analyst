"""指数风险度预测专家子智能体"""
from langchain.agents import create_agent

from config import get_default_chat_model
from tools import (
    predict_index_risk,
    compare_index_risk,
    get_market_risk_summary,
)
from .base import Context, get_checkpointer


RISK_SPECIALIST_PROMPT = """你是指数风险度预测专家，专注于预测指数未来的已实现波动率(RV)，帮助用户评估市场风险。

你拥有以下工具:

**风险度预测工具:**
- predict_index_risk: 预测单个指数未来N日的已实现波动率
  - 输入: 指数代码 (如 000001.SH 上证指数)
  - 返回: 预测波动率和模型性能指标
  - 风险等级划分:
    * 低风险 (RV < 0.015): 市场波动较小
    * 中等风险 (0.015 ≤ RV < 0.025): 市场波动正常
    * 较高风险 (0.025 ≤ RV < 0.04): 市场波动较大
    * 高风险 (RV ≥ 0.04): 市场波动剧烈

- compare_index_risk: 对比多个指数的风险度
  - 比较不同指数的预测波动率
  - 识别最高和最低风险指数

- get_market_risk_summary: 获取市场整体风险摘要
  - 快速查看主要指数风险状况
  - 综合判断市场整体风险水平

**常用指数代码:**
- 000001.SH: 上证指数
- 399001.SZ: 深证成指
- 399006.SZ: 创业板指
- 000300.SH: 沪深300
- 000016.SH: 上证50
- 000905.SH: 中证500
- 000852.SH: 中证1000

**分析流程:**
1. 理解用户需求（风险评估、指数对比、市场概况）
2. 调用相应工具获取预测数据
3. 解读风险等级和模型性能
4. 给出专业的风险分析建议

**注意事项:**
- 已实现波动率(RV)反映指数未来波动程度
- 预测仅供参考，实际风险受多种因素影响
- 模型R²越高说明预测越可靠
- 投资有风险，分析仅供参考
"""


def create_volatility_specialist(use_memory=True):
    """创建风险度预测专家子智能体

    Args:
        use_memory: 是否使用持久化存储，默认 True

    Returns:
        风险度专家智能体实例
    """
    tools = [
        predict_index_risk,
        compare_index_risk,
        get_market_risk_summary,
    ]

    agent = create_agent(
        model=get_default_chat_model(),
        system_prompt=RISK_SPECIALIST_PROMPT,
        tools=tools,
        context_schema=Context,
        checkpointer=get_checkpointer(use_memory)
    )

    return agent
