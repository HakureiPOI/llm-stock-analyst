"""
验证工具模块 - 输入验证和数据校验
"""
import re
from typing import Optional, List, Tuple
from datetime import datetime


# ============ 股票代码验证 ============

def validate_stock_code(code: str) -> Tuple[bool, str]:
    """
    验证股票代码格式
    
    Args:
        code: 股票代码，如 '600519.SH'
        
    Returns:
        (是否有效，错误信息)
    """
    if not code:
        return False, "股票代码不能为空"
    
    # 标准格式：6 位数字 + . + 交易所 (SH/SZ)
    pattern = r'^\d{6}\.(SH|SZ)$'
    if not re.match(pattern, code):
        return False, f"股票代码格式错误：{code}，应为 6 位数字+.SH 或.SZ，如 600519.SH"
    
    return True, ""


def validate_index_code(code: str) -> Tuple[bool, str]:
    """
    验证指数代码格式
    
    Args:
        code: 指数代码，如 '000001.SH'
        
    Returns:
        (是否有效，错误信息)
    """
    if not code:
        return False, "指数代码不能为空"
    
    # 指数格式：6 位数字 + . + 交易所 (SH/SZ)
    pattern = r'^\d{6}\.(SH|SZ)$'
    if not re.match(pattern, code):
        return False, f"指数代码格式错误：{code}"
    
    return True, ""


def parse_stock_codes(codes_str: str) -> Tuple[List[str], str]:
    """
    解析逗号分隔的多个股票代码
    
    Args:
        codes_str: 逗号分隔的股票代码字符串
        
    Returns:
        (代码列表，错误信息)
    """
    if not codes_str:
        return [], "股票代码字符串不能为空"
    
    codes = [c.strip() for c in codes_str.split(',') if c.strip()]
    if not codes:
        return [], "未找到有效的股票代码"
    
    invalid_codes = []
    for code in codes:
        valid, err = validate_stock_code(code)
        if not valid:
            invalid_codes.append(code)
    
    if invalid_codes:
        return codes, f"以下股票代码格式错误：{', '.join(invalid_codes)}"
    
    return codes, ""


# ============ 日期验证 ============

def validate_date(date_str: str, field_name: str = "日期") -> Tuple[bool, str]:
    """
    验证日期格式 (YYYYMMDD)
    
    Args:
        date_str: 日期字符串
        field_name: 字段名称（用于错误信息）
        
    Returns:
        (是否有效，错误信息)
    """
    if not date_str:
        return True, ""  # 空日期视为有效（可选参数）
    
    if not re.match(r'^\d{8}$', date_str):
        return False, f"{field_name}格式错误：{date_str}，应为 YYYYMMDD"
    
    try:
        datetime.strptime(date_str, '%Y%m%d')
        return True, ""
    except ValueError:
        return False, f"{field_name}无效：{date_str}"


def validate_date_range(
    start_date: Optional[str],
    end_date: Optional[str]
) -> Tuple[bool, str]:
    """
    验证日期范围
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        (是否有效，错误信息)
    """
    # 验证格式
    valid, err = validate_date(start_date, "开始日期")
    if not valid:
        return False, err
    
    valid, err = validate_date(end_date, "结束日期")
    if not valid:
        return False, err
    
    # 验证范围
    if start_date and end_date:
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        if start > end:
            return False, "开始日期不能晚于结束日期"
    
    return True, ""


def validate_trade_date(trade_date: str) -> Tuple[bool, str]:
    """
    验证交易日期格式
    
    Args:
        trade_date: 交易日期
        
    Returns:
        (是否有效，错误信息)
    """
    return validate_date(trade_date, "交易日期")


# ============ 数值验证 ============

def validate_positive_int(value: Optional[int], field_name: str) -> Tuple[bool, str]:
    """
    验证正整数
    
    Args:
        value: 整数值
        field_name: 字段名称
        
    Returns:
        (是否有效，错误信息)
    """
    if value is None:
        return True, ""  # 可选参数
    
    if not isinstance(value, int):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return False, f"{field_name}必须是整数"
    
    if value <= 0:
        return False, f"{field_name}必须是正整数"
    
    return True, ""


def validate_limit_offset(
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> Tuple[bool, str]:
    """
    验证 limit 和 offset 参数
    
    Returns:
        (是否有效，错误信息)
    """
    valid, err = validate_positive_int(limit, "limit")
    if not valid:
        return False, err
    
    valid, err = validate_positive_int(offset, "offset")
    if not valid:
        return False, err
    
    return True, ""


# ============ 指数代码白名单 ============

SUPPORTED_INDICES = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000300.SH": "沪深 300",
    "000016.SH": "上证 50",
    "000905.SH": "中证 500",
    "000852.SH": "中证 1000",
}


def validate_supported_index(code: str) -> Tuple[bool, str]:
    """
    验证是否为支持的指数
    
    Args:
        code: 指数代码
        
    Returns:
        (是否有效，错误信息)
    """
    valid, err = validate_index_code(code)
    if not valid:
        return False, err
    
    if code not in SUPPORTED_INDICES:
        supported_list = ", ".join(SUPPORTED_INDICES.keys())
        return False, f"不支持的指数：{code}，支持的指数：{supported_list}"
    
    return True, ""


# ============ 综合验证装饰器 ============

from functools import wraps


def validate_params(validation_func):
    """
    参数验证装饰器
    
    Usage:
        @validate_params(lambda ts_code, **kwargs: validate_stock_code(ts_code))
        def get_stock_info(ts_code: str, ...):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取参数
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            
            # 执行验证
            valid, err = validation_func(**bound.arguments)
            if not valid:
                return f"参数验证失败：{err}"
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
