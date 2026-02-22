import hashlib
from cachetools import TTLCache
from functools import wraps

# 创建全局缓存实例
_data_cache = TTLCache(maxsize=1000, ttl=3600)  # 1小时TTL


def cached_data(*args_keys):
    """
    数据缓存装饰器

    Args:
        *args_keys: 需要参与生成缓存键的参数索引列表（从0开始）
                   如果不指定，则所有参数都参与生成键

    Usage:
        @cached_data()  # 所有参数都参与
        def get_data(param1, param2=""):
            ...

        @cached_data(0, 2)  # 只有第0和第2个参数参与
        def get_data(param1, param2="", param3=""):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 生成缓存键 - 包含kwargs
            method_name = func.__name__

            # 处理位置参数
            if args_keys:
                key_args = [str(args[i]) if i < len(args) else "" for i in args_keys]
            else:
                key_args = [str(arg) if arg else "" for arg in args]

            # 处理关键字参数并排序以确保一致性
            if kwargs:
                sorted_kwargs = sorted(kwargs.items())
                key_args.extend([f"{k}={v}" for k, v in sorted_kwargs])

            key_str = f"{method_name}:{':'.join(key_args)}"
            cache_key = hashlib.md5(key_str.encode()).hexdigest()

            # 检查缓存
            if cache_key in _data_cache:
                return _data_cache[cache_key]

            # 调用原始方法
            result = func(self, *args, **kwargs)

            # 存入缓存
            _data_cache[cache_key] = result
            return result

        return wrapper
    return decorator


def clear_cache():
    """清空所有缓存"""
    _data_cache.clear()


def get_cache_info():
    """获取缓存信息"""
    return {
        "size": len(_data_cache),
        "maxsize": _data_cache.maxsize,
        "ttl": _data_cache.ttl
    }
