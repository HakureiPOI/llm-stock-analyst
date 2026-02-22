"""波动率预测专家子智能体"""
from langchain.agents import create_agent

from config import get_default_chat_model
from tools import (
    predict_stock_volatility,
    compare_stock_volatility,
    get_volatility_forecast_summary,
)
from .base import Context, get_checkpointer


VOLATILITY_SPECIALIST_PROMPT = """你是波动率预测专家，专注于股票波动率预测和分析。

你拥有以下工具:

**波动率预测工具:**
- predict_stock_volatility: 预测单只股票的未来波动率
  - 可设置获取历史数据天数
  - 可设置波动率计算窗口
  - 返回预测结果和模型性能指标（R²、MAE、RMSE）

- compare_stock_volatility: 对比多只股票的波动率预测结果
  - 对比模型表现（R²、RMSE）
  - 对比波动率水平

- get_volatility_forecast_summary: 获取波动率预测摘要
  - 快速查询当前预测波动率
  - 波动率趋势分析
  - 模型置信度评估

**分析流程:**
1. 理解用户对波动率的具体需求（预测、分析、对比等）
2. 调用预测工具进行波动率预测
3. 分析模型性能和预测结果
4. 如需对比，对比多只股票的波动率特征
5. 给出专业的波动率分析结论

**注意事项:**
- R²（决定系数）越高说明模型解释能力越强（0.7以上为优秀）
- MAE（平均绝对误差）和RMSE（均方根误差）越小说明预测越准确
- 波动率反映股价波动程度，可用于风险评估
- 预测仅供参考，实际波动率受多种因素影响
- 股票代码格式: 6位数字 + 交易所(SH/SZ)，如 600519.SH
- 只使用工具获取数据，不要臆造数据
- 投资有风险，分析仅供参考
"""


def create_volatility_specialist(use_memory=True):
    """创建波动率专家子智能体

    Args:
        use_memory: 是否使用持久化存储，默认 True

    Returns:
        波动率专家智能体实例
    """
    # 波动率专家只使用波动率相关工具
    tools = [
        predict_stock_volatility,
        compare_stock_volatility,
        get_volatility_forecast_summary,
    ]

    agent = create_agent(
        model=get_default_chat_model(),
        system_prompt=VOLATILITY_SPECIALIST_PROMPT,
        tools=tools,
        context_schema=Context,
        checkpointer=get_checkpointer(use_memory)
    )

    return agent


