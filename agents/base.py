"""智能体基础模块 - 上下文和共享配置"""
import os
import sys
from pathlib import Path
import dotenv
from dataclasses import dataclass
from typing import Optional
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

dotenv.load_dotenv()

DB_URI = os.getenv("DB_URI")


@dataclass
class Context:
    """自定义运行时上下文"""
    user_id: str = '1'
    session_id: Optional[str] = None


def get_checkpointer(use_memory=True):
    """获取检查点保存器
    
    Args:
        use_memory: 是否使用持久化存储（PostgreSQL），否则使用内存
    
    Returns:
        检查点保存器实例
    """
    if use_memory and DB_URI:
        # PostgresSaver.from_conn_string 返回的是上下文管理器，需要进入
        checkpointer = PostgresSaver.from_conn_string(DB_URI).__enter__()
    else:
        checkpointer = InMemorySaver()
    return checkpointer
