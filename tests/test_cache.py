"""
缓存模块单元测试
"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tsdata.cache import (
    _get_cache_key,
    cached_data,
    clear_cache,
    get_cache_info,
    cache_stats,
)


class TestCacheKeyGeneration:
    """测试缓存键生成"""
    
    def test_simple_args(self):
        """测试简单参数"""
        key = _get_cache_key("test_method", ("arg1", "arg2"), {})
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 长度
    
    def test_with_kwargs(self):
        """测试关键字参数"""
        key1 = _get_cache_key("test_method", ("arg1",), {"kw1": "value1"})
        key2 = _get_cache_key("test_method", ("arg1",), {"kw1": "value1"})
        assert key1 == key2  # 相同参数应该生成相同键
    
    def test_kwargs_order_independent(self):
        """测试关键字参数顺序无关"""
        key1 = _get_cache_key("test_method", (), {"a": "1", "b": "2"})
        key2 = _get_cache_key("test_method", (), {"b": "2", "a": "1"})
        assert key1 == key2
    
    def test_none_vs_empty_string(self):
        """测试 None 和空字符串的区别"""
        key1 = _get_cache_key("test_method", (None,), {})
        key2 = _get_cache_key("test_method", ("",), {})
        assert key1 != key2  # 应该不同
    
    def test_list_args(self):
        """测试列表参数"""
        key1 = _get_cache_key("test_method", (["a", "b"],), {})
        key2 = _get_cache_key("test_method", (["a", "b"],), {})
        assert key1 == key2
    
    def test_different_list_order(self):
        """测试不同顺序的列表"""
        key1 = _get_cache_key("test_method", (["a", "b"],), {})
        key2 = _get_cache_key("test_method", (["b", "a"],), {})
        assert key1 != key2


class TestCacheDecorator:
    """测试缓存装饰器"""
    
    def setup_method(self):
        """每个测试前清空缓存"""
        clear_cache()
    
    def test_caching_basic(self):
        """测试基本缓存功能"""
        call_count = [0]
        
        class TestClass:
            @cached_data()
            def get_data(self, value):
                call_count[0] += 1
                return f"data_{value}"
        
        obj = TestClass()
        
        # 第一次调用
        result1 = obj.get_data("test")
        assert result1 == "data_test"
        assert call_count[0] == 1
        
        # 第二次调用（应该命中缓存）
        result2 = obj.get_data("test")
        assert result2 == "data_test"
        assert call_count[0] == 1  # 不应该增加
    
    def test_different_args_different_cache(self):
        """测试不同参数使用不同缓存"""
        call_count = [0]
        
        class TestClass:
            @cached_data()
            def get_data(self, value):
                call_count[0] += 1
                return f"data_{value}"
        
        obj = TestClass()
        
        result1 = obj.get_data("test1")
        result2 = obj.get_data("test2")
        
        assert call_count[0] == 2  # 应该调用两次
        assert result1 == "data_test1"
        assert result2 == "data_test2"
    
    def test_selected_args_keys(self):
        """测试指定参数键"""
        call_count = [0]
        
        class TestClass:
            @cached_data(0)  # 只使用第一个参数
            def get_data(self, param1, param2="default"):
                call_count[0] += 1
                return f"{param1}_{param2}"
        
        obj = TestClass()
        
        # 相同 param1，不同 param2
        result1 = obj.get_data("test", "value1")
        result2 = obj.get_data("test", "value2")
        
        # 因为只使用 param1 作为键，所以应该命中缓存
        assert call_count[0] == 1
        assert result1 == result2


class TestCacheOperations:
    """测试缓存操作"""
    
    def setup_method(self):
        """每个测试前清空缓存"""
        clear_cache()
    
    def test_clear_cache(self):
        """测试清空缓存"""
        # 先添加一些数据
        class TestClass:
            @cached_data()
            def get_data(self, value):
                return f"data_{value}"
        
        obj = TestClass()
        obj.get_data("test1")
        obj.get_data("test2")
        
        info_before = get_cache_info()
        assert info_before["size"] == 2
        
        # 清空缓存
        count = clear_cache()
        assert count == 2
        
        info_after = get_cache_info()
        assert info_after["size"] == 0
    
    def test_get_cache_info(self):
        """测试获取缓存信息"""
        info = get_cache_info()
        
        assert "size" in info
        assert "maxsize" in info
        assert "ttl" in info
        assert info["maxsize"] == 500
        assert info["ttl"] == 3600
    
    def test_cache_stats(self):
        """测试缓存统计"""
        stats = cache_stats()
        
        assert "entries" in stats
        assert "max_entries" in stats
        assert "ttl_seconds" in stats
        assert "size_mb" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
