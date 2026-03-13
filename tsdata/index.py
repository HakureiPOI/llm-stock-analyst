"""指数数据获取模块"""
from typing import Optional
import pandas as pd

from .client import pro, TushareClient
from .cache import cached_data
from utils.validators import validate_index_code, validate_date_range
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _safe_call(func, method_name: str, **kwargs) -> pd.DataFrame:
    """
    安全调用 Tushare API，带限流和重试
    
    Args:
        func: Tushare API 函数
        method_name: 方法名称（用于日志）
        **kwargs: API 参数
        
    Returns:
        DataFrame 结果
    """
    try:
        return TushareClient.rate_limited_call(func, **kwargs)
    except Exception as e:
        logger.error(f"{method_name} 失败：{e}")
        return pd.DataFrame()


class IndexData:
    """指数数据获取类"""

    @cached_data()
    def get_index_basic(
        self,
        ts_code: str = "",
        name: str = "",
        market: str = "",
        publisher: str = "",
        category: str = "",
        limit: str = "",
        offset: str = ""
    ) -> pd.DataFrame:
        """
        获取指数基本信息
        """
        # 验证指数代码格式（如果提供）
        if ts_code:
            codes = [c.strip() for c in ts_code.split(',') if c.strip()]
            for code in codes:
                valid, err = validate_index_code(code)
                if not valid:
                    logger.warning(f"指数代码格式可能有问题：{err}")
        
        logger.debug(f"获取指数基础信息：ts_code={ts_code or 'all'}")
        
        try:
            return _safe_call(
                pro.index_basic, "get_index_basic", **{
                    "ts_code": ts_code,
                    "name": name,
                    "market": market,
                    "publisher": publisher,
                    "category": category,
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "ts_code", "name", "market", "publisher", "category",
                    "base_date", "base_point", "list_date", "fullname",
                    "index_type", "weight_rule", "desc", "exp_date"
                ]
            )
        except Exception as e:
            logger.error(f"获取指数基础信息失败：{e}")
            return pd.DataFrame()

    @cached_data()
    def get_index_daily(
        self,
        ts_code: str,
        trade_date: str = "",
        start_date: str = "",
        end_date: str = "",
        limit: str = "",
        offset: str = ""
    ) -> pd.DataFrame:
        """
        获取指数日 K 线数据
        """
        # 验证指数代码
        if not ts_code:
            logger.error("ts_code 为必填参数")
            return pd.DataFrame()
        
        valid, err = validate_index_code(ts_code)
        if not valid:
            logger.error(f"指数代码验证失败：{err}")
            return pd.DataFrame()
        
        # 验证日期范围
        valid, err = validate_date_range(start_date, end_date)
        if not valid:
            logger.error(f"日期验证失败：{err}")
            return pd.DataFrame()
        
        logger.debug(f"获取指数日 K 线：ts_code={ts_code}, start={start_date or 'N/A'}, end={end_date or 'N/A'}")
        
        try:
            return _safe_call(
                pro.index_daily, "get_index_daily", **{
                    "ts_code": ts_code,
                    "trade_date": trade_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "close", "open", "high", "low",
                    "pre_close", "change", "pct_chg", "vol", "amount"
                ]
            )
        except Exception as e:
            logger.error(f"获取指数日 K 线失败：{e}")
            return pd.DataFrame()

    @cached_data()
    def get_index_weight(
        self,
        index_code: str = "",
        trade_date: str = "",
        start_date: str = "",
        end_date: str = "",
        ts_code: str = "",
        limit: str = "",
        offset: str = ""
    ) -> pd.DataFrame:
        """
        获取指数成分股权重数据
        """
        if index_code:
            valid, err = validate_index_code(index_code)
            if not valid:
                logger.error(f"指数代码验证失败：{err}")
                return pd.DataFrame()
        
        logger.debug(f"获取指数权重：index_code={index_code}, trade_date={trade_date}")
        
        try:
            return _safe_call(
                pro.index_weight, "get_index_weight", **{
                    "index_code": index_code,
                    "trade_date": trade_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "ts_code": ts_code,
                    "limit": limit,
                    "offset": offset
                }, fields=["index_code", "con_code", "trade_date", "weight"]
            )
        except Exception as e:
            logger.error(f"获取指数权重失败：{e}")
            return pd.DataFrame()

    @cached_data()
    def get_index_dailybasic(
        self,
        trade_date: str = "",
        ts_code: str = "",
        start_date: str = "",
        end_date: str = "",
        limit: str = "",
        offset: str = ""
    ) -> pd.DataFrame:
        """
        获取指数每日基本面指标数据
        """
        if ts_code:
            valid, err = validate_index_code(ts_code)
            if not valid:
                logger.error(f"指数代码验证失败：{err}")
                return pd.DataFrame()
        
        # 验证日期范围
        valid, err = validate_date_range(start_date, end_date)
        if not valid:
            logger.error(f"日期验证失败：{err}")
            return pd.DataFrame()
        
        logger.debug(f"获取指数基本面数据：ts_code={ts_code}")
        
        try:
            return _safe_call(
                pro.index_dailybasic, "get_index_dailybasic", **{
                    "trade_date": trade_date,
                    "ts_code": ts_code,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "total_mv", "float_mv", "total_share",
                    "float_share", "free_share", "turnover_rate", "turnover_rate_f",
                    "pe", "pe_ttm", "pb"
                ]
            )
        except Exception as e:
            logger.error(f"获取指数基本面数据失败：{e}")
            return pd.DataFrame()


# 创建全局实例
index_data = IndexData()
