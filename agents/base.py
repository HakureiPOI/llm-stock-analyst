"""智能体基础模块 - 上下文和共享配置"""
import os
import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import dotenv
from dataclasses import dataclass
from contextlib import contextmanager

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置日志
from utils.logger import setup_logger
logger = setup_logger(__name__)

dotenv.load_dotenv()

DB_URI = os.getenv("DB_URI")


@dataclass
class Context:
    """自定义运行时上下文"""
    user_id: str = '1'
    session_id: Optional[str] = None


class CheckpointerManager:
    """检查点保存器管理器 - 管理 PostgreSQL 连接生命周期"""
    
    _instance = None
    _checkpointer = None
    _use_postgres = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CheckpointerManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, use_memory: bool = True):
        """初始化检查点保存器"""
        if cls._checkpointer is not None:
            return cls._checkpointer
        
        if use_memory and DB_URI:
            try:
                from langgraph.checkpoint.postgres import PostgresSaver
                # 使用上下文管理器正确管理连接
                cls._checkpointer = PostgresSaver.from_conn_string(DB_URI)
                cls._use_postgres = True
                logger.info("PostgreSQL 检查点保存器初始化成功")
            except Exception as e:
                logger.warning(f"PostgreSQL 初始化失败，回退到内存存储：{e}")
                from langgraph.checkpoint.memory import InMemorySaver
                cls._checkpointer = InMemorySaver()
                cls._use_postgres = False
        else:
            from langgraph.checkpoint.memory import InMemorySaver
            cls._checkpointer = InMemorySaver()
            cls._use_postgres = False
            logger.info("内存检查点保存器初始化成功")
        
        return cls._checkpointer
    
    @classmethod
    def get_checkpointer(cls):
        """获取检查点保存器实例"""
        if cls._checkpointer is None:
            return cls.initialize()
        return cls._checkpointer
    
    @classmethod
    def close(cls):
        """关闭 PostgreSQL 连接（如果有）"""
        if cls._checkpointer is not None and cls._use_postgres:
            try:
                # PostgresSaver 有 close 方法
                if hasattr(cls._checkpointer, 'close'):
                    cls._checkpointer.close()
                logger.info("PostgreSQL 连接已关闭")
            except Exception as e:
                logger.error(f"关闭 PostgreSQL 连接时出错：{e}")
            finally:
                cls._checkpointer = None
                cls._use_postgres = False


@contextmanager
def checkpointer_context(use_memory: bool = True):
    """
    检查点保存器上下文管理器
    
    Usage:
        with checkpointer_context(use_memory=True) as checkpointer:
            agent = create_agent(..., checkpointer=checkpointer)
    """
    checkpointer = CheckpointerManager.initialize(use_memory)
    try:
        yield checkpointer
    finally:
        # 注意：LangGraph 会管理检查点的生命周期，这里不主动关闭
        pass


def get_checkpointer(use_memory: bool = True):
    """
    获取检查点保存器（兼容旧接口）
    
    Args:
        use_memory: 是否使用持久化存储（PostgreSQL），否则使用内存

    Returns:
        检查点保存器实例
        
    Note: 推荐使用 CheckpointerManager 类来管理生命周期
    """
    return CheckpointerManager.get_checkpointer()


def cleanup_checkpointer():
    """清理检查点保存器资源"""
    CheckpointerManager.close()
