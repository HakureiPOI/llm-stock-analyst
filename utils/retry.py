"""
重试和限流工具模块
"""
import time
from functools import wraps
from typing import Optional, Tuple, Type, Union
import logging

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (Exception,),
    logger_name: Optional[str] = None
):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 延迟倍增系数
        exceptions: 需要重试的异常类型
        logger_name: 日志记录器名称
        
    Usage:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def api_call():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logging.getLogger(logger_name or func.__module__)
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        _logger.warning(
                            f"{func.__name__} 失败 (尝试 {attempt + 1}/{max_attempts}): {e}. "
                            f"{current_delay:.1f}秒后重试..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        _logger.error(
                            f"{func.__name__} 失败，已达最大重试次数 {max_attempts}: {e}"
                        )
            
            # 抛出最后一次异常
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def rate_limit(calls: int = 1, period: float = 1.0):
    """
    限流装饰器
    
    Args:
        calls: 允许调用次数
        period: 时间窗口（秒）
        
    Usage:
        @rate_limit(calls=1, period=2.0)  # 每 2 秒最多调用 1 次
        def api_call():
            ...
    """
    def decorator(func):
        # 存储上次调用时间
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            elapsed = current_time - last_called[0]
            
            if elapsed < period:
                wait_time = period - elapsed
                logger.debug(f"{func.__name__} 触发限流，等待 {wait_time:.2f}秒")
                time.sleep(wait_time)
            
            last_called[0] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit_with_key(key_func, calls: int = 1, period: float = 1.0):
    """
    基于键的限流装饰器（支持多个键独立限流）
    
    Args:
        key_func: 生成限流键的函数
        calls: 每个键允许的调用次数
        period: 时间窗口（秒）
        
    Usage:
        @rate_limit_with_key(lambda ts_code: ts_code, calls=5, period=60)
        def get_stock_data(ts_code):
            ...
    """
    def decorator(func):
        # 存储每个键的最后调用时间
        last_called = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取限流键
            key = key_func(*args, **kwargs)
            current_time = time.time()
            
            if key in last_called:
                elapsed = current_time - last_called[key]
                if elapsed < period:
                    wait_time = period - elapsed
                    logger.debug(f"{func.__name__}[{key}] 触发限流，等待 {wait_time:.2f}秒")
                    time.sleep(wait_time)
            
            last_called[key] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class RateLimiter:
    """
    限流器类（支持更复杂的限流场景）
    """
    
    def __init__(self, calls: int, period: float):
        """
        Args:
            calls: 允许调用次数
            period: 时间窗口（秒）
        """
        self.calls = calls
        self.period = period
        self._timestamps = []
    
    def acquire(self) -> bool:
        """
        获取调用许可
        
        Returns:
            是否成功获取（总是返回 True，但可能会阻塞）
        """
        import time
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
                logger.debug(f"限流中，等待 {wait_time:.2f}秒")
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
