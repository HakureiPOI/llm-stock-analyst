"""指数数据查询工具"""
from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import tool
from tsdata import index_data


class IndexBasicQuery(BaseModel):
    ts_code: Optional[str] = Field(None, description="指数代码，支持逗号分隔的多个代码，如 '000001.SH,000300.SH'")
    name: Optional[str] = Field(None, description="指数名称，如 '上证指数'")
    market: Optional[str] = Field(None, description="市场，如 'SSE'、'SZSE'")
    category: Optional[str] = Field(None, description="指数类别，如 '规模指数'、'行业指数'、'主题指数'")
    limit: Optional[int] = Field(None, description="返回记录数量限制")


class IndexDailyQuery(BaseModel):
    ts_code: str = Field(..., description="指数代码，如 '000001.SH'")
    start_date: Optional[str] = Field(None, description="开始日期，格式：YYYYMMDD")
    end_date: Optional[str] = Field(None, description="结束日期，格式：YYYYMMDD")
    limit: Optional[int] = Field(None, description="返回记录数量限制")


class IndexWeightQuery(BaseModel):
    index_code: Optional[str] = Field(None, description="指数代码，如 '000001.SH'")
    trade_date: Optional[str] = Field(None, description="交易日期，格式：YYYYMMDD")
    start_date: Optional[str] = Field(None, description="开始日期")
    end_date: Optional[str] = Field(None, description="结束日期")
    limit: Optional[int] = Field(None, description="返回记录数量限制")


@tool(args_schema=IndexBasicQuery)
def get_index_basic_info(ts_code=None, name=None, market=None, category=None, limit=None) -> str:
    """查询指数基础信息，包括指数代码、名称、市场、基点、上市日期等。"""
    df = index_data.get_index_basic(
        ts_code=ts_code or "",
        name=name or "",
        market=market or "",
        publisher="",
        category=category or "",
        limit=str(limit) if limit else "",
        offset=""
    )
    if df.empty:
        return "未找到匹配的指数信息。"
    return df.to_json(orient="records", force_ascii=False, indent=2)


@tool(args_schema=IndexDailyQuery)
def get_index_daily_kline(ts_code: str, start_date=None, end_date=None, limit=None) -> str:
    """获取指数日K线数据，包括开盘价、最高价、最低价、收盘价、成交量、成交额等。"""
    df = index_data.get_index_daily(
        ts_code=ts_code,
        trade_date="",
        start_date=start_date or "",
        end_date=end_date or "",
        limit=str(limit) if limit else "",
        offset=""
    )
    if df.empty:
        return "未找到该指数的K线数据。"
    return df.to_json(orient="records", force_ascii=False, indent=2)


@tool(args_schema=IndexWeightQuery)
def get_index_weight(index_code=None, trade_date=None, start_date=None, end_date=None, limit=None) -> str:
    """获取指数成分股权重数据，查看指数中各股票的权重占比。"""
    df = index_data.get_index_weight(
        index_code=index_code or "",
        trade_date=trade_date or "",
        start_date=start_date or "",
        end_date=end_date or "",
        ts_code="",
        limit=str(limit) if limit else "",
        offset=""
    )
    if df.empty:
        return "未找到该指数的成分股权重数据。"
    return df.to_json(orient="records", force_ascii=False, indent=2)


@tool
def get_index_dailybasic(trade_date=None, ts_code=None, start_date=None, end_date=None, limit=None) -> str:
    """获取指数每日基本面指标数据，包括总市值、流通市值、PE、PB等指标。"""
    df = index_data.get_index_dailybasic(
        trade_date=trade_date or "",
        ts_code=ts_code or "",
        start_date=start_date or "",
        end_date=end_date or "",
        limit=str(limit) if limit else "",
        offset=""
    )
    if df.empty:
        return "未找到该指数的基本面数据。"
    return df.to_json(orient="records", force_ascii=False, indent=2)
