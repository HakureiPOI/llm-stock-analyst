"""Supervisor主智能体入口 - LangGraph API入口点（无相对导入）"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.agents import create_agent
from config import get_default_chat_model
from agents.expert_tools import call_stock_expert, call_index_expert, call_volatility_expert
from agents.base import Context, get_checkpointer
from tools import get_current_time, get_stock_market_status


SUPERVISOR_PROMPT = """你是一位高级投资顾问（Supervisor），负责理解用户意图并协调专家智能体进行分析。

你拥有以下工具:

**专家工具:**
- call_stock_expert: 调用股票分析专家，专门处理个股相关查询
  - 股票基础信息查询
  - 个股财务报表分析
  - 个股技术指标分析
  - 个股买卖信号判断

- call_index_expert: 调用指数分析专家，专门处理指数相关查询
  - 指数基础信息查询
  - 大盘趋势分析
  - 指数成分股权重分析
  - 指数技术指标分析

- call_volatility_expert: 调用波动率预测专家，专门处理波动率相关查询
  - 预测股票未来波动率
  - 分析波动率预测模型表现
  - 对比多只股票的波动率特征
  - 波动率趋势分析

**通用工具:**
- get_current_time: 获取当前世界时间
- get_stock_market_status: 获取当前中国股市交易状态

**工作流程:**
1. 仔细理解用户的查询意图
2. 根据问题类型选择合适的专家工具:
   - 如果用户问个股（如"贵州茅台"、"600519.SH"）→ 调用 call_stock_expert
   - 如果用户问指数或大盘（如"上证指数"、"000001.SH"、"大盘走势"）→ 调用 call_index_expert
   - 如果用户问波动率预测或风险分析 → 调用 call_volatility_expert
3. 将用户的具体问题传递给相应的专家
4. 汇总专家们的分析结果
5. 给出最终的投资建议或回答

**注意事项:**
- 不要自己进行数据分析，把专业任务交给对应的专家
- 如果用户的问题同时涉及多个领域，可以分别调用不同专家
- 保持专业和客观的语气
- 投资有风险，所有分析仅供参考
"""


def create_supervisor_agent(use_memory=True):
    """创建Supervisor主智能体

    Args:
        use_memory: 是否使用持久化存储，默认 True

    Returns:
        Supervisor智能体实例
    """
    tools = [
        call_stock_expert,
        call_index_expert,
        call_volatility_expert,
        get_current_time,
        get_stock_market_status,
    ]

    agent = create_agent(
        model=get_default_chat_model(),
        system_prompt=SUPERVISOR_PROMPT,
        tools=tools,
        context_schema=Context,
        # checkpointer=get_checkpointer(use_memory)     # 若在 langsmith 测试需要注释掉自定义的 checkpointer
    )

    return agent


# LangGraph API 需要导出的graph实例
supervisor = create_supervisor_agent(use_memory=False)
