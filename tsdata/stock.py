"""股票数据获取模块"""
from typing import Optional, List, Callable, Any
import pandas as pd

from .client import pro, TushareClient
from .cache import cached_data
from utils.validators import validate_stock_code, validate_date_range
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _safe_call(func: Callable, method_name: str, **kwargs) -> pd.DataFrame:
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


class StockData:
    """股票数据获取类"""

    @cached_data()
    def get_stock_basic(
        self,
        ts_code: str = "",
        name: str = "",
        exchange: str = "",
        market: str = "",
        is_hs: str = "",
        list_status: str = "",
        limit: str = "",
        offset: str = ""
    ) -> pd.DataFrame:
        """
        获取股票基本信息

        Args:
            ts_code: 股票代码，支持逗号分隔的多个代码
            name: 股票名称
            exchange: 交易所代码
            market: 市场类型
            is_hs: 是否沪深港通标的
            list_status: 上市状态
            limit: 限制返回数量
            offset: 偏移量

        Returns:
            DataFrame: 股票基本信息数据
        """
        # 验证股票代码格式（如果提供）
        if ts_code:
            codes = [c.strip() for c in ts_code.split(',') if c.strip()]
            for code in codes:
                valid, err = validate_stock_code(code)
                if not valid:
                    logger.warning(f"股票代码格式可能有问题：{err}")
        
        logger.debug(f"获取股票基础信息：ts_code={ts_code or 'all'}")
        
        try:
            return _safe_call(
                pro.stock_basic, "get_stock_basic", **{
                    "ts_code": ts_code,
                    "name": name,
                    "exchange": exchange,
                    "market": market,
                    "is_hs": is_hs,
                    "list_status": list_status,
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "ts_code", "symbol", "name", "area", "industry", "cnspell",
                    "market", "list_date", "act_name", "act_ent_type", "fullname",
                    "enname", "exchange", "curr_type", "list_status", "delist_date", "is_hs"
                ]
            )
        except Exception as e:
            logger.error(f"获取股票基础信息失败：{e}")
            return pd.DataFrame()

    @cached_data()
    def get_daily(
        self,
        ts_code: str = "",
        trade_date: str = "",
        start_date: str = "",
        end_date: str = "",
        limit: str = "",
        offset: str = ""
    ) -> pd.DataFrame:
        """
        获取股票日 K 线数据

        Args:
            ts_code: 股票代码，支持逗号分隔的多个代码
            trade_date: 交易日期，YYYYMMDD 格式
            start_date: 开始日期，YYYYMMDD 格式
            end_date: 结束日期，YYYYMMDD 格式
            limit: 限制返回数量
            offset: 偏移量

        Returns:
            DataFrame: 股票日 K 线数据
        """
        # 验证股票代码
        if ts_code:
            valid, err = validate_stock_code(ts_code.split(',')[0])
            if not valid:
                logger.error(f"股票代码验证失败：{err}")
                return pd.DataFrame()

        # 验证日期范围
        valid, err = validate_date_range(start_date, end_date)
        if not valid:
            logger.error(f"日期验证失败：{err}")
            return pd.DataFrame()

        logger.debug(f"获取日 K 线数据：ts_code={ts_code}, start={start_date or 'N/A'}, end={end_date or 'N/A'}")

        try:
            return _safe_call(
                pro.daily, "get_daily", **{
                    "ts_code": ts_code,
                    "trade_date": trade_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "open", "high", "low", "close",
                    "pre_close", "change", "pct_chg", "vol", "amount"
                ]
            )
        except Exception as e:
            logger.error(f"获取日 K 线数据失败：{e}")
            return pd.DataFrame()

    @cached_data()
    def get_adj_factor(
        self,
        ts_code: str = "",
        trade_date: str = "",
        start_date: str = "",
        end_date: str = "",
        limit: str = "",
        offset: str = ""
    ) -> pd.DataFrame:
        """
        获取股票复权因子数据

        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            limit: 限制返回数量
            offset: 偏移量

        Returns:
            DataFrame: 复权因子数据
        """
        if ts_code:
            valid, err = validate_stock_code(ts_code.split(',')[0])
            if not valid:
                logger.error(f"股票代码验证失败：{err}")
                return pd.DataFrame()

        logger.debug(f"获取复权因子：ts_code={ts_code}")

        try:
            return _safe_call(
                pro.adj_factor, "get_adj_factor", **{
                    "ts_code": ts_code,
                    "trade_date": trade_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                    "offset": offset
                }, fields=["ts_code", "trade_date", "adj_factor"]
            )
        except Exception as e:
            logger.error(f"获取复权因子失败：{e}")
            return pd.DataFrame()

    @cached_data()
    def get_daily_basic(
        self,
        ts_code: str = "",
        trade_date: str = "",
        start_date: str = "",
        end_date: str = "",
        limit: str = "",
        offset: str = ""
    ) -> pd.DataFrame:
        """
        获取股票每日基本面指标数据

        Args:
            ts_code: 股票代码
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            limit: 限制返回数量
            offset: 偏移量

        Returns:
            DataFrame: 每日基本面指标数据
        """
        if ts_code:
            valid, err = validate_stock_code(ts_code.split(',')[0])
            if not valid:
                logger.error(f"股票代码验证失败：{err}")
                return pd.DataFrame()
        
        logger.debug(f"获取每日基本面数据：ts_code={ts_code}")

        try:
            return _safe_call(
                pro.daily_basic, "get_daily_basic", **{
                    "ts_code": ts_code,
                    "trade_date": trade_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f",
                    "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio",
                    "dv_ttm", "total_share", "float_share", "free_share", "total_mv",
                    "circ_mv", "limit_status"
                ]
            )
        except Exception as e:
            logger.error(f"获取每日基本面数据失败：{e}")
            return pd.DataFrame()

    @cached_data()
    def get_income(
        self,
        ts_code: str,
        ann_date: str = "",
        f_ann_date: str = "",
        start_date: str = "",
        end_date: str = "",
        period: str = "",
        report_type: str = "",
        comp_type: str = "",
        is_calc: str = "",
        limit: str = "",
        offset: str = ""
    ) -> pd.DataFrame:
        """
        获取股票利润表数据

        Args:
            ts_code: 股票代码（必填）
            ann_date: 公告日期
            f_ann_date: 实际公告日期
            start_date: 开始日期
            end_date: 结束日期
            period: 报告期
            report_type: 报表类型
            comp_type: 公司类型
            is_calc: 是否计算
            limit: 限制返回数量
            offset: 偏移量

        Returns:
            DataFrame: 利润表数据
        """
        # 验证必填参数
        if not ts_code:
            logger.error("ts_code 为必填参数")
            return pd.DataFrame()
        
        valid, err = validate_stock_code(ts_code.split(',')[0])
        if not valid:
            logger.error(f"股票代码验证失败：{err}")
            return pd.DataFrame()
        
        logger.debug(f"获取利润表：ts_code={ts_code}, period={period}")

        try:
            return _safe_call(
                pro.income, "get_income", **{
                    "ts_code": ts_code,
                    "ann_date": ann_date,
                    "f_ann_date": f_ann_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "period": period,
                    "report_type": report_type,
                    "comp_type": comp_type,
                    "is_calc": is_calc,
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "ts_code", "ann_date", "f_ann_date", "end_date", "report_type",
                    "comp_type", "end_type", "basic_eps", "diluted_eps", "total_revenue",
                    "revenue", "oper_cost", "oper_profit", "total_profit", "n_income",
                    "n_income_attr_p", "ebit", "ebitda", "rd_exp"
                ]
            )
        except Exception as e:
            logger.error(f"获取利润表失败：{e}")
            return pd.DataFrame()

    @cached_data()
    def get_balancesheet(
        self,
        ts_code: str,
        ann_date: str = "",
        f_ann_date: str = "",
        start_date: str = "",
        end_date: str = "",
        period: str = "",
        report_type: str = "",
        comp_type: str = "",
        limit: str = "",
        offset: str = ""
    ) -> pd.DataFrame:
        """
        获取股票资产负债表数据

        Args:
            ts_code: 股票代码（必填）
            ann_date: 公告日期
            start_date: 开始日期
            end_date: 结束日期
            period: 报告期
            report_type: 报表类型
            comp_type: 公司类型
            limit: 限制返回数量
            offset: 偏移量

        Returns:
            DataFrame: 资产负债表数据
        """
        if not ts_code:
            logger.error("ts_code 为必填参数")
            return pd.DataFrame()
        
        valid, err = validate_stock_code(ts_code.split(',')[0])
        if not valid:
            logger.error(f"股票代码验证失败：{err}")
            return pd.DataFrame()
        
        logger.debug(f"获取资产负债表：ts_code={ts_code}")

        try:
            return _safe_call(
                pro.balancesheet, "get_balancesheet", **{
                    "ts_code": ts_code,
                    "ann_date": ann_date,
                    "f_ann_date": f_ann_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "period": period,
                    "report_type": report_type,
                    "comp_type": comp_type,
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "ts_code", "ann_date", "end_date", "total_assets", "total_liab",
                    "total_hldr_eqy_exc_min_int", "money_cap", "accounts_receiv",
                    "inventories", "fix_assets", "total_cur_assets", "total_cur_liab"
                ]
            )
        except Exception as e:
            logger.error(f"获取资产负债表失败：{e}")
            return pd.DataFrame()

    @cached_data()
    def get_cashflow(
        self,
        ts_code: str,
        ann_date: str = "",
        f_ann_date: str = "",
        start_date: str = "",
        end_date: str = "",
        period: str = "",
        report_type: str = "",
        comp_type: str = "",
        is_calc: str = "",
        limit: str = "",
        offset: str = ""
    ) -> pd.DataFrame:
        """
        获取股票现金流量表数据

        Args:
            ts_code: 股票代码（必填）
            ann_date: 公告日期
            start_date: 开始日期
            end_date: 结束日期
            period: 报告期
            report_type: 报表类型
            comp_type: 公司类型
            is_calc: 是否计算
            limit: 限制返回数量
            offset: 偏移量

        Returns:
            DataFrame: 现金流量表数据
        """
        if not ts_code:
            logger.error("ts_code 为必填参数")
            return pd.DataFrame()
        
        valid, err = validate_stock_code(ts_code.split(',')[0])
        if not valid:
            logger.error(f"股票代码验证失败：{err}")
            return pd.DataFrame()
        
        logger.debug(f"获取现金流量表：ts_code={ts_code}")

        try:
            return _safe_call(
                pro.cashflow, "get_cashflow", **{
                    "ts_code": ts_code,
                    "ann_date": ann_date,
                    "f_ann_date": f_ann_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "period": period,
                    "report_type": report_type,
                    "comp_type": comp_type,
                    "is_calc": is_calc,
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "ts_code", "ann_date", "end_date", "net_profit", "n_cashflow_act",
                    "n_cashflow_inv_act", "n_cash_flows_fnc_act", "n_incr_cash_cash_equ",
                    "free_cashflow"
                ]
            )
        except Exception as e:
            logger.error(f"获取现金流量表失败：{e}")
            return pd.DataFrame()

    @cached_data()
    def get_fina_indicator(
        self,
        ts_code: str,
        ann_date: str = "",
        start_date: str = "",
        end_date: str = "",
        period: str = "",
        update_flag: str = "",
        limit: str = "",
        offset: str = ""
    ) -> pd.DataFrame:
        """
        获取股票财务指标数据

        Args:
            ts_code: 股票代码（必填）
            ann_date: 公告日期
            start_date: 开始日期
            end_date: 结束日期
            period: 报告期
            update_flag: 更新标志
            limit: 限制返回数量
            offset: 偏移量

        Returns:
            DataFrame: 财务指标数据
        """
        if not ts_code:
            logger.error("ts_code 为必填参数")
            return pd.DataFrame()
        
        valid, err = validate_stock_code(ts_code.split(',')[0])
        if not valid:
            logger.error(f"股票代码验证失败：{err}")
            return pd.DataFrame()
        
        logger.debug(f"获取财务指标：ts_code={ts_code}")

        try:
            return _safe_call(
                pro.fina_indicator, "get_fina_indicator", **{
                    "ts_code": ts_code,
                    "ann_date": ann_date,
                    "start_date": start_date,
                    "end_date": end_date,
                    "period": period,
                    "update_flag": update_flag,
                    "limit": limit,
                    "offset": offset
                }, fields=[
                    "ts_code", "ann_date", "end_date", "eps", "roe", "roa", "gross_margin",
                    "netprofit_margin", "current_ratio", "quick_ratio", "debt_to_assets",
                    "ar_turn", "ca_turn", "fa_turn", "assets_turn"
                ]
            )
        except Exception as e:
            logger.error(f"获取财务指标失败：{e}")
            return pd.DataFrame()


# 创建全局实例
stock_data = StockData()
