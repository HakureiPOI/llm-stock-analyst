"""股票数据查询工具"""
from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import tool
import json

from tsdata import stock_data
from utils.validators import validate_stock_code, validate_date_range, validate_positive_int
from utils.logger import setup_logger

logger = setup_logger(__name__)


class StockBasicQuery(BaseModel):
    ts_code: Optional[str] = Field(None, description="股票代码，支持逗号分隔的多个代码，如 '600000.SH,000001.SZ'")
    name: Optional[str] = Field(None, description="股票名称，如 '平安银行'")
    exchange: Optional[str] = Field(None, description="交易所代码，如 'SSE' 或 'SZSE'")
    market: Optional[str] = Field(None, description="市场类型，如 '主板'、'创业板'、'科创板'")
    industry: Optional[str] = Field(None, description="所属行业，如 '银行'、'科技'")
    limit: Optional[int] = Field(None, description="返回记录数量限制，默认返回所有")


class StockDailyQuery(BaseModel):
    ts_code: str = Field(..., description="股票代码，如 '600000.SH'")
    start_date: Optional[str] = Field(None, description="开始日期，格式：YYYYMMDD，如 '20240101'")
    end_date: Optional[str] = Field(None, description="结束日期，格式：YYYYMMDD，如 '20241231'")
    limit: Optional[int] = Field(None, description="返回记录数量限制")


class StockFinancialQuery(BaseModel):
    ts_code: str = Field(..., description="股票代码，如 '600000.SH'")
    period: Optional[str] = Field(None, description="报告期，如 '20241231'")
    limit: Optional[int] = Field(None, description="返回记录数量限制")


@tool(args_schema=StockBasicQuery)
def get_stock_basic_info(
    ts_code: Optional[str] = None,
    name: Optional[str] = None,
    exchange: Optional[str] = None,
    market: Optional[str] = None,
    industry: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """查询股票基础信息，包括股票代码、名称、行业、上市日期、交易所等。"""
    try:
        # 验证股票代码（如果提供）
        if ts_code:
            codes = [c.strip() for c in ts_code.split(',') if c.strip()]
            for code in codes:
                valid, err = validate_stock_code(code)
                if not valid:
                    return json.dumps({"error": f"股票代码验证失败：{err}"}, ensure_ascii=False)
        
        df = stock_data.get_stock_basic(
            ts_code=ts_code or "",
            name=name or "",
            exchange=exchange or "",
            market=market or "",
            limit=str(limit) if limit else "",
            offset=""
        )
        
        if df.empty:
            return json.dumps({"error": "未找到匹配的股票信息"}, ensure_ascii=False)
        
        return df.to_json(orient="records", force_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"get_stock_basic_info 失败：{e}")
        return json.dumps({"error": f"查询失败：{str(e)}"}, ensure_ascii=False)


@tool(args_schema=StockDailyQuery)
def get_stock_daily_kline(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """获取股票日 K 线数据，包括开盘价、最高价、最低价、收盘价、成交量、成交额等。"""
    try:
        # 验证股票代码
        valid, err = validate_stock_code(ts_code)
        if not valid:
            return json.dumps({"error": f"股票代码验证失败：{err}"}, ensure_ascii=False)
        
        # 验证日期范围
        valid, err = validate_date_range(start_date, end_date)
        if not valid:
            return json.dumps({"error": f"日期验证失败：{err}"}, ensure_ascii=False)
        
        df = stock_data.get_daily(
            ts_code=ts_code,
            trade_date="",
            start_date=start_date or "",
            end_date=end_date or "",
            limit=str(limit) if limit else "",
            offset=""
        )
        
        if df.empty:
            return json.dumps({"error": "未找到该股票的 K 线数据"}, ensure_ascii=False)
        
        return df.to_json(orient="records", force_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"get_stock_daily_kline 失败：{e}")
        return json.dumps({"error": f"查询失败：{str(e)}"}, ensure_ascii=False)


@tool
def get_stock_financial_data(
    ts_code: str,
    trade_date: str = "",
    start_date: str = "",
    end_date: str = "",
    limit: int = 5
) -> str:
    """获取股票的每日基本面数据，包括市值、市盈率、市净率、换手率等。"""
    try:
        # 验证股票代码
        valid, err = validate_stock_code(ts_code)
        if not valid:
            return json.dumps({"error": f"股票代码验证失败：{err}"}, ensure_ascii=False)
        
        df = stock_data.get_daily_basic(
            ts_code=ts_code,
            trade_date=trade_date or "",
            start_date=start_date or "",
            end_date=end_date or "",
            limit=str(limit),
            offset=""
        )
        
        if df.empty:
            return json.dumps({"error": "未找到该股票的基本面数据"}, ensure_ascii=False)
        
        return df.to_json(orient="records", force_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"get_stock_financial_data 失败：{e}")
        return json.dumps({"error": f"查询失败：{str(e)}"}, ensure_ascii=False)


@tool(args_schema=StockFinancialQuery)
def get_stock_income(
    ts_code: str,
    period: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """获取股票利润表数据，包括营业收入、净利润、EPS 等财务指标。"""
    try:
        # 验证股票代码
        valid, err = validate_stock_code(ts_code)
        if not valid:
            return json.dumps({"error": f"股票代码验证失败：{err}"}, ensure_ascii=False)
        
        df = stock_data.get_income(
            ts_code=ts_code,
            period=period or "",
            limit=str(limit) if limit else "",
            offset=""
        )
        
        if df.empty:
            return json.dumps({"error": "未找到该股票的利润表数据"}, ensure_ascii=False)
        
        return df.to_json(orient="records", force_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"get_stock_income 失败：{e}")
        return json.dumps({"error": f"查询失败：{str(e)}"}, ensure_ascii=False)


@tool(args_schema=StockFinancialQuery)
def get_stock_balance_sheet(
    ts_code: str,
    period: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """获取股票资产负债表数据，包括总资产、总负债、净资产等。"""
    try:
        # 验证股票代码
        valid, err = validate_stock_code(ts_code)
        if not valid:
            return json.dumps({"error": f"股票代码验证失败：{err}"}, ensure_ascii=False)
        
        df = stock_data.get_balancesheet(
            ts_code=ts_code,
            period=period or "",
            limit=str(limit) if limit else "",
            offset=""
        )
        
        if df.empty:
            return json.dumps({"error": "未找到该股票的资产负债表数据"}, ensure_ascii=False)
        
        return df.to_json(orient="records", force_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"get_stock_balance_sheet 失败：{e}")
        return json.dumps({"error": f"查询失败：{str(e)}"}, ensure_ascii=False)


@tool(args_schema=StockFinancialQuery)
def get_stock_cashflow(
    ts_code: str,
    period: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """获取股票现金流量表数据，包括经营活动现金流、投资活动现金流等。"""
    try:
        # 验证股票代码
        valid, err = validate_stock_code(ts_code)
        if not valid:
            return json.dumps({"error": f"股票代码验证失败：{err}"}, ensure_ascii=False)
        
        df = stock_data.get_cashflow(
            ts_code=ts_code,
            period=period or "",
            limit=str(limit) if limit else "",
            offset=""
        )
        
        if df.empty:
            return json.dumps({"error": "未找到该股票的现金流量表数据"}, ensure_ascii=False)
        
        return df.to_json(orient="records", force_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"get_stock_cashflow 失败：{e}")
        return json.dumps({"error": f"查询失败：{str(e)}"}, ensure_ascii=False)


@tool(args_schema=StockFinancialQuery)
def get_stock_fina_indicator(
    ts_code: str,
    period: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """获取股票财务指标数据，包括 ROE、ROA、资产负债率、毛利率等关键指标。"""
    try:
        # 验证股票代码
        valid, err = validate_stock_code(ts_code)
        if not valid:
            return json.dumps({"error": f"股票代码验证失败：{err}"}, ensure_ascii=False)
        
        df = stock_data.get_fina_indicator(
            ts_code=ts_code,
            period=period or "",
            limit=str(limit) if limit else "",
            offset=""
        )
        
        if df.empty:
            return json.dumps({"error": "未找到该股票的财务指标数据"}, ensure_ascii=False)
        
        return df.to_json(orient="records", force_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"get_stock_fina_indicator 失败：{e}")
        return json.dumps({"error": f"查询失败：{str(e)}"}, ensure_ascii=False)


@tool
def get_adj_factor(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """获取股票复权因子数据，用于计算复权后的价格。"""
    try:
        # 验证股票代码
        valid, err = validate_stock_code(ts_code)
        if not valid:
            return json.dumps({"error": f"股票代码验证失败：{err}"}, ensure_ascii=False)
        
        # 验证日期范围
        valid, err = validate_date_range(start_date, end_date)
        if not valid:
            return json.dumps({"error": f"日期验证失败：{err}"}, ensure_ascii=False)
        
        df = stock_data.get_adj_factor(
            ts_code=ts_code,
            start_date=start_date or "",
            end_date=end_date or "",
            limit=str(limit) if limit else "",
            offset=""
        )
        
        if df.empty:
            return json.dumps({"error": "未找到该股票的复权因子数据"}, ensure_ascii=False)
        
        return df.to_json(orient="records", force_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"get_adj_factor 失败：{e}")
        return json.dumps({"error": f"查询失败：{str(e)}"}, ensure_ascii=False)
