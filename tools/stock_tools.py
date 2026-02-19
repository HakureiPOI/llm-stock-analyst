"""股票数据查询工具"""
from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import tool
from tsdata import stock_data, get_cache_info


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
def get_stock_basic_info(ts_code=None, name=None, exchange=None, market=None, industry=None, limit=None) -> str:
    """查询股票基础信息，包括股票代码、名称、行业、上市日期、交易所等。"""
    df = stock_data.get_stock_basic(
        ts_code=ts_code or "",
        name=name or "",
        exchange=exchange or "",
        market=market or "",
        limit=str(limit) if limit else "",
        offset=""
    )
    if df.empty:
        return "未找到匹配的股票信息。"
    return df.to_json(orient="records", force_ascii=False, indent=2)


@tool(args_schema=StockDailyQuery)
def get_stock_daily_kline(ts_code: str, start_date=None, end_date=None, limit=None) -> str:
    """获取股票日K线数据，包括开盘价、最高价、最低价、收盘价、成交量、成交额等。"""
    df = stock_data.get_daily(
        ts_code=ts_code,
        trade_date="",
        start_date=start_date or "",
        end_date=end_date or "",
        limit=str(limit) if limit else "",
        offset=""
    )
    if df.empty:
        return "未找到该股票的K线数据。"
    return df.to_json(orient="records", force_ascii=False, indent=2)


@tool
def get_stock_financial_data(ts_code: str, period: str = "", limit: int = 5) -> str:
    """获取股票的每日基本面数据，包括市值、市盈率、市净率、换手率等。"""
    df = stock_data.get_daily_basic(
        ts_code=ts_code,
        trade_date="",
        start_date="",
        end_date="",
        limit=str(limit),
        offset=""
    )
    if df.empty:
        return "未找到该股票的基本面数据。"
    return df.to_json(orient="records", force_ascii=False, indent=2)


@tool(args_schema=StockFinancialQuery)
def get_stock_income(ts_code: str, period=None, limit=None) -> str:
    """获取股票利润表数据，包括营业收入、净利润、EPS等财务指标。"""
    df = stock_data.get_income(
        ts_code=ts_code,
        ann_date="",
        f_ann_date="",
        start_date="",
        end_date="",
        period=period or "",
        report_type="",
        comp_type="",
        is_calc="",
        limit=str(limit) if limit else "",
        offset=""
    )
    if df.empty:
        return "未找到该股票的利润表数据。"
    return df.to_json(orient="records", force_ascii=False, indent=2)


@tool(args_schema=StockFinancialQuery)
def get_stock_balance_sheet(ts_code: str, period=None, limit=None) -> str:
    """获取股票资产负债表数据，包括总资产、总负债、净资产等。"""
    df = stock_data.get_balancesheet(
        ts_code=ts_code,
        ann_date="",
        f_ann_date="",
        start_date="",
        end_date="",
        period=period or "",
        report_type="",
        comp_type="",
        limit=str(limit) if limit else "",
        offset=""
    )
    if df.empty:
        return "未找到该股票的资产负债表数据。"
    return df.to_json(orient="records", force_ascii=False, indent=2)


@tool(args_schema=StockFinancialQuery)
def get_stock_cashflow(ts_code: str, period=None, limit=None) -> str:
    """获取股票现金流量表数据，包括经营活动现金流、投资活动现金流等。"""
    df = stock_data.get_cashflow(
        ts_code=ts_code,
        ann_date="",
        f_ann_date="",
        start_date="",
        end_date="",
        period=period or "",
        report_type="",
        comp_type="",
        is_calc="",
        limit=str(limit) if limit else "",
        offset=""
    )
    if df.empty:
        return "未找到该股票的现金流量表数据。"
    return df.to_json(orient="records", force_ascii=False, indent=2)


@tool(args_schema=StockFinancialQuery)
def get_stock_fina_indicator(ts_code: str, period=None, limit=None) -> str:
    """获取股票财务指标数据，包括ROE、ROA、资产负债率、毛利率等关键指标。"""
    df = stock_data.get_fina_indicator(
        ts_code=ts_code,
        ann_date="",
        start_date="",
        end_date="",
        period=period or "",
        update_flag="",
        limit=str(limit) if limit else "",
        offset=""
    )
    if df.empty:
        return "未找到该股票的财务指标数据。"
    return df.to_json(orient="records", force_ascii=False, indent=2)


@tool
def get_adj_factor(ts_code: str, start_date=None, end_date=None, limit=None) -> str:
    """获取股票复权因子数据，用于计算复权后的价格。"""
    df = stock_data.get_adj_factor(
        ts_code=ts_code,
        trade_date="",
        start_date=start_date or "",
        end_date=end_date or "",
        limit=str(limit) if limit else "",
        offset=""
    )
    if df.empty:
        return "未找到该股票的复权因子数据。"
    return df.to_json(orient="records", force_ascii=False, indent=2)


@tool
def get_cache_status() -> str:
    """查看当前数据缓存状态，包括缓存数量和配置。"""
    info = get_cache_info()
    return f"""
数据缓存状态：
- 当前缓存数量: {info['size']}
- 最大缓存数量: {info['maxsize']}
- 缓存过期时间: {info['ttl']} 秒
"""
