"""
数据缓存模块 - 支持 TTL 的数据缓存
"""
import hashlib
import json
from typing import Any, List, Optional, Callable
from functools import wraps
from cachetools import TTLCache
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)

# 创建全局缓存实例
_data_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)  # 1 小时 TTL，最大 500 条目


def _get_cache_key(method_name: str, args: tuple, kwargs: dict) -> str:
    """
    生成缓存键，避免键碰撞
    
    Args:
        method_name: 方法名
        args: 位置参数
        kwargs: 关键字参数
        
    Returns:
        MD5 哈希的缓存键
    """
    # 使用特殊标记区分 None 和空字符串
    NONE_MARKER = "__NONE__"
    EMPTY_MARKER = "__EMPTY__"
    
    def normalize_value(val: Any) -> str:
        """规范化值用于缓存键生成"""
        if val is None:
            return NONE_MARKER
        elif val == "":
            return EMPTY_MARKER
        elif isinstance(val, (list, tuple)):
            return f"[{','.join(normalize_value(v) for v in val)}]"
        elif isinstance(val, dict):
            sorted_items = sorted(val.items())
            return f"{{{','.join(f'{k}:{normalize_value(v)}' for k, v in sorted_items)}}}"
        else:
            return str(val) if val else EMPTY_MARKER
    
    # 处理位置参数
    args_part = [normalize_value(arg) for arg in args]
    
    # 处理关键字参数（排序以确保一致性）
    kwargs_part = [f"{k}={normalize_value(v)}" for k, v in sorted(kwargs.items())]
    
    # 组合键
    key_parts = [method_name] + args_part + kwargs_part
    key_str = ":".join(key_parts)
    
    # 生成 MD5 哈希
    cache_key = hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    return cache_key


def cached_data(*args_keys: int):
    """
    数据缓存装饰器

    Args:
        *args_keys: 需要参与生成缓存键的参数索引列表（从 0 开始）
                   如果不指定，则所有参数都参与生成键

    Usage:
        @cached_data()  # 所有参数都参与
        def get_data(param1, param2=""):
            ...

        @cached_data(0, 2)  # 只有第 0 和第 2 个参数参与
        def get_data(param1, param2="", param3=""):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 生成缓存键
            method_name = func.__name__
            
            # 如果指定了参数键，只使用这些参数
            if args_keys:
                filtered_args = tuple(
                    args[i] if i < len(args) else None 
                    for i in args_keys
                )
            else:
                filtered_args = args
            
            cache_key = _get_cache_key(method_name, filtered_args, kwargs)
            
            # 检查缓存
            if cache_key in _data_cache:
                logger.debug(f"缓存命中：{method_name} - {cache_key[:8]}...")
                return _data_cache[cache_key]
            
            # 调用原始方法
            try:
                result = func(self, *args, **kwargs)
                
                # 检查返回值是否适合缓存
                if result is not None:
                    # 对于 DataFrame，检查大小
                    if isinstance(result, pd.DataFrame):
                        size_mb = result.memory_usage(deep=True).sum() / (1024 * 1024)
                        if size_mb > 50:  # 大于 50MB 不缓存
                            logger.warning(
                                f"数据过大 ({size_mb:.1f}MB)，不缓存：{method_name}"
                            )
                            return result
                    
                    # 存入缓存
                    _data_cache[cache_key] = result
                    logger.debug(f"缓存已设置：{method_name} - {cache_key[:8]}...")
                
                return result
                
            except Exception as e:
                logger.error(f"缓存方法执行失败 {method_name}: {e}")
                raise
        
        return wrapper
    return decorator


def clear_cache(method_name: Optional[str] = None) -> int:
    """
    清空缓存
    
    Args:
        method_name: 可选，只清空特定方法的缓存
        
    Returns:
        清除的缓存条目数量
    """
    if method_name:
        # 只清除特定方法的缓存
        keys_to_delete = [
            key for key in _data_cache.keys()
            if key.startswith(hashlib.md5(method_name.encode()).hexdigest()[:8])
        ]
        for key in keys_to_delete:
            del _data_cache[key]
        logger.info(f"已清除 {len(keys_to_delete)} 条 {method_name} 的缓存")
        return len(keys_to_delete)
    else:
        # 清除所有缓存
        count = len(_data_cache)
        _data_cache.clear()
        logger.info(f"已清除所有缓存 ({count}条)")
        return count


def get_cache_info() -> dict:
    """获取缓存信息"""
    return {
        "size": len(_data_cache),
        "maxsize": _data_cache.maxsize,
        "ttl": _data_cache.ttl,
        "currsize_bytes": sum(
            len(str(v).encode('utf-8')) for v in _data_cache.values()
        ),
    }


def cache_stats() -> dict:
    """获取缓存统计数据"""
    info = get_cache_info()
    return {
        "entries": info["size"],
        "max_entries": info["maxsize"],
        "ttl_seconds": info["ttl"],
        "size_mb": info["currsize_bytes"] / (1024 * 1024),
    }
