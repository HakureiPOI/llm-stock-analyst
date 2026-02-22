"""智能体工厂 - 统一的智能体创建接口"""
from .stock_specialist import create_stock_specialist
from .index_specialist import create_index_specialist
from .volatility_specialist import create_volatility_specialist
from .supervisor_entry import create_supervisor_agent
from .base import Context


def create_agent(agent_type: str, use_memory: bool = True, **kwargs):
    """创建指定类型的智能体

    Args:
        agent_type: 智能体类型，可选值:
            - "supervisor": Supervisor主智能体（推荐）
            - "stock": 股票专家
            - "index": 指数专家
            - "volatility": 波动率专家
        use_memory: 是否使用持久化存储，默认 True
        **kwargs: 其他参数

    Returns:
        智能体实例

    Raises:
        ValueError: 当agent_type不支持时
    """
    agent_map = {
        "supervisor": create_supervisor_agent,
        "stock": create_stock_specialist,
        "index": create_index_specialist,
        "volatility": create_volatility_specialist,
    }

    if agent_type not in agent_map:
        raise ValueError(
            f"不支持的智能体类型: {agent_type}。"
            f"支持的类型: {list(agent_map.keys())}"
        )

    return agent_map[agent_type](use_memory=use_memory)


__all__ = [
    "create_agent",
    "create_supervisor_agent",
    "create_stock_specialist",
    "create_index_specialist",
    "create_volatility_specialist",
    "Context",
]
