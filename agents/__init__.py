# 智能体模块（多智能体架构）
from .agent_factory import (
    create_agent,
    create_supervisor_agent,
    create_stock_specialist,
    create_index_specialist,
    create_volatility_specialist,
    Context,
)

__all__ = [
    "create_agent",                # 统一创建接口
    "create_supervisor_agent",     # Supervisor主智能体
    "create_stock_specialist",     # 股票专家
    "create_index_specialist",     # 指数专家
    "create_volatility_specialist", # 波动率专家
    "Context",                     # 上下文
]
