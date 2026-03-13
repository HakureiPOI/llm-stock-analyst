"""
验证工具单元测试
"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.validators import (
    validate_stock_code,
    validate_index_code,
    validate_date,
    validate_date_range,
    validate_positive_int,
    validate_supported_index,
    parse_stock_codes,
    SUPPORTED_INDICES,
)


class TestValidateStockCode:
    """测试股票代码验证"""
    
    def test_valid_sh_stock(self):
        """测试有效的上海股票代码"""
        valid, err = validate_stock_code("600519.SH")
        assert valid is True
        assert err == ""
    
    def test_valid_sz_stock(self):
        """测试有效的深圳股票代码"""
        valid, err = validate_stock_code("000001.SZ")
        assert valid is True
        assert err == ""
    
    def test_invalid_format(self):
        """测试无效格式"""
        valid, err = validate_stock_code("600519")
        assert valid is False
        assert "格式错误" in err
    
    def test_invalid_exchange(self):
        """测试无效的交易所"""
        valid, err = validate_stock_code("600519.SS")
        assert valid is False
        assert "格式错误" in err
    
    def test_empty_code(self):
        """测试空代码"""
        valid, err = validate_stock_code("")
        assert valid is False
        assert "不能为空" in err
    
    def test_none_code(self):
        """测试 None"""
        valid, err = validate_stock_code(None)
        assert valid is False


class TestValidateIndexCode:
    """测试指数代码验证"""
    
    def test_valid_sh_index(self):
        """测试有效的上海指数"""
        valid, err = validate_index_code("000001.SH")
        assert valid is True
    
    def test_valid_sz_index(self):
        """测试有效的深圳指数"""
        valid, err = validate_index_code("399001.SZ")
        assert valid is True
    
    def test_invalid_format(self):
        """测试无效格式"""
        valid, err = validate_index_code("000001")
        assert valid is False


class TestValidateDate:
    """测试日期验证"""
    
    def test_valid_date(self):
        """测试有效日期"""
        valid, err = validate_date("20240101")
        assert valid is True
    
    def test_invalid_format(self):
        """测试无效格式"""
        valid, err = validate_date("2024-01-01")
        assert valid is False
        assert "格式错误" in err
    
    def test_invalid_date(self):
        """测试无效日期"""
        valid, err = validate_date("20240230")
        assert valid is False
    
    def test_empty_date(self):
        """测试空日期（可选参数）"""
        valid, err = validate_date("")
        assert valid is True


class TestValidateDateRange:
    """测试日期范围验证"""
    
    def test_valid_range(self):
        """测试有效范围"""
        valid, err = validate_date_range("20240101", "20241231")
        assert valid is True
    
    def test_start_after_end(self):
        """测试开始日期晚于结束日期"""
        valid, err = validate_date_range("20241231", "20240101")
        assert valid is False
        assert "开始日期不能晚于结束日期" in err
    
    def test_both_empty(self):
        """测试两个日期都为空"""
        valid, err = validate_date_range(None, None)
        assert valid is True


class TestValidatePositiveInt:
    """测试正整数验证"""
    
    def test_valid_positive(self):
        """测试有效的正整数"""
        valid, err = validate_positive_int(10, "limit")
        assert valid is True
    
    def test_zero(self):
        """测试零"""
        valid, err = validate_positive_int(0, "limit")
        assert valid is False
        assert "必须是正整数" in err
    
    def test_negative(self):
        """测试负数"""
        valid, err = validate_positive_int(-5, "limit")
        assert valid is False
    
    def test_none_value(self):
        """测试 None（可选参数）"""
        valid, err = validate_positive_int(None, "limit")
        assert valid is True


class TestParseStockCodes:
    """测试多个股票代码解析"""
    
    def test_single_code(self):
        """测试单个代码"""
        codes, err = parse_stock_codes("600519.SH")
        assert len(codes) == 1
        assert codes[0] == "600519.SH"
        assert err == ""
    
    def test_multiple_codes(self):
        """测试多个代码"""
        codes, err = parse_stock_codes("600519.SH,000001.SZ,300750.SZ")
        assert len(codes) == 3
        assert err == ""
    
    def test_with_spaces(self):
        """测试带空格的输入"""
        codes, err = parse_stock_codes("600519.SH, 000001.SZ , 300750.SZ")
        assert len(codes) == 3
        assert "600519.SH" in codes
        assert "000001.SZ" in codes
        assert "300750.SZ" in codes
    
    def test_invalid_code_in_list(self):
        """测试列表中有无效代码"""
        codes, err = parse_stock_codes("600519.SH,invalid,000001.SZ")
        assert len(codes) == 3
        assert "invalid" in err
    
    def test_empty_string(self):
        """测试空字符串"""
        codes, err = parse_stock_codes("")
        assert len(codes) == 0
        assert err != ""


class TestValidateSupportedIndex:
    """测试支持的指数验证"""
    
    def test_supported_index(self):
        """测试支持的指数"""
        valid, err = validate_supported_index("000001.SH")
        assert valid is True
    
    def test_valid_but_not_supported(self):
        """测试格式正确但不在白名单的指数"""
        valid, err = validate_supported_index("000002.SH")
        assert valid is False
        assert "不支持的指数" in err
    
    def test_all_supported_indices(self):
        """测试所有支持的指数"""
        for code in SUPPORTED_INDICES.keys():
            valid, err = validate_supported_index(code)
            assert valid is True, f"{code} 应该被支持"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
