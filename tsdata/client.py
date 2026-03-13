import tushare as ts
from dotenv import load_dotenv
import os
from typing import Optional
from functools import wraps
import time

from utils.logger import setup_logger
from utils.retry import retry

logger = setup_logger(__name__)

load_dotenv()

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')

# 启动时验证必要的 API Token
if not TUSHARE_TOKEN:
    raise ValueError(
        "TUSHARE_TOKEN 环境变量未设置，请在 .env 文件中配置。\n"
        "示例：TUSHARE_TOKEN=your_tushare_token_here"
    )


# Tushare API 限流器 - 根据 Tushare 的限流规则配置
# 积分等级限流规则:
#   0 分：20 次/分钟
#   100 分：50 次/分钟
#   500 分：100 次/分钟
#   1000 分：200 次/分钟
#   2000 分：400 次/分钟
#   5000 分 +：500+ 次/分钟
# 当前配置：2000 积分 = 400 次/分钟 ≈ 6.7 次/秒
class TushareRateLimiter:
    """Tushare API 限流器"""

    def __init__(self, calls: int = 400, period: float = 60.0):
        """
        Args:
            calls: 允许调用次数
            period: 时间窗口（秒）
        """
        self.calls = calls
        self.period = period
        self._timestamps = []
    
    def acquire(self) -> bool:
        """获取调用许可"""
        current_time = time.time()
        
        # 移除过期的时间戳
        self._timestamps = [
            ts for ts in self._timestamps
            if current_time - ts < self.period
        ]
        
        # 如果达到限制，等待
        if len(self._timestamps) >= self.calls:
            wait_time = self.period - (current_time - self._timestamps[0])
            if wait_time > 0:
                logger.debug(f"Tushare API 限流中，等待 {wait_time:.2f}秒")
                time.sleep(wait_time)
                # 重新清理过期时间戳
                current_time = time.time()
                self._timestamps = [
                    ts for ts in self._timestamps
                    if current_time - ts < self.period
                ]
        
        # 记录本次调用
        self._timestamps.append(current_time)
        return True
    
    def __call__(self, func):
        """作为装饰器使用"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)
        return wrapper


# 全局限流器实例
# 2000 积分 = 400 次/分钟
_tushare_rate_limiter = TushareRateLimiter(calls=400, period=60.0)


class TushareClient:
    _instance: Optional['TushareClient'] = None
    pro: Optional[ts.pro_api] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TushareClient, cls).__new__(cls)
            try:
                cls._instance.pro = ts.pro_api(TUSHARE_TOKEN)
            except Exception as e:
                raise RuntimeError(f"初始化 Tushare 客户端失败：{e}")
        return cls._instance
    
    @staticmethod
    def rate_limited_call(func, *args, **kwargs):
        """
        带限流和重试的 API 调用
        
        Args:
            func: 要调用的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数调用结果
        """
        # 应用限流
        _tushare_rate_limiter.acquire()
        
        # 应用重试
        last_exception = None
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < 2:  # 前两次失败才重试
                    wait_time = 2 ** attempt  # 指数退避：1s, 2s
                    logger.warning(
                        f"Tushare API 调用失败 (尝试 {attempt + 1}/3): {e}. "
                        f"{wait_time}秒后重试..."
                    )
                    time.sleep(wait_time)
        
        # 所有重试失败
        if last_exception:
            logger.error(f"Tushare API 调用失败，已达最大重试次数：{last_exception}")
            raise last_exception


client = TushareClient()
pro = client.pro
