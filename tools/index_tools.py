"""指数数据查询工具"""
from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import tool
import json

from tsdata import index_data
from utils.validators import validate_index_code, validate_date_range
from utils.logger import setup_logger

logger = setup_logger(__name__)


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
def get_index_basic_info(
    ts_code: Optional[str] = None,
    name: Optional[str] = None,
    market: Optional[str] = None,
    category: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """查询指数基础信息，包括指数代码、名称、市场、基点、上市日期等。"""
    try:
        # 验证指数代码（如果提供）
        if ts_code:
            codes = [c.strip() for c in ts_code.split(',') if c.strip()]
            for code in codes:
                valid, err = validate_index_code(code)
                if not valid:
                    return json.dumps({"error": f"指数代码验证失败：{err}"}, ensure_ascii=False)
        
        df = index_data.get_index_basic(
            ts_code=ts_code or "",
            name=name or "",
            market=market or "",
            category=category or "",
            limit=str(limit) if limit else "",
            offset=""
        )
        
        if df.empty:
            return json.dumps({"error": "未找到匹配的指数信息"}, ensure_ascii=False)
        
        return df.to_json(orient="records", force_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"get_index_basic_info 失败：{e}")
        return json.dumps({"error": f"查询失败：{str(e)}"}, ensure_ascii=False)


@tool(args_schema=IndexDailyQuery)
def get_index_daily_kline(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """获取指数日 K 线数据，包括开盘价、最高价、最低价、收盘价、成交量、成交额等。"""
    try:
        # 验证指数代码
        valid, err = validate_index_code(ts_code)
        if not valid:
            return json.dumps({"error": f"指数代码验证失败：{err}"}, ensure_ascii=False)
        
        # 验证日期范围
        valid, err = validate_date_range(start_date, end_date)
        if not valid:
            return json.dumps({"error": f"日期验证失败：{err}"}, ensure_ascii=False)
        
        df = index_data.get_index_daily(
            ts_code=ts_code,
            start_date=start_date or "",
            end_date=end_date or "",
            limit=str(limit) if limit else "",
            offset=""
        )
        
        if df.empty:
            return json.dumps({"error": "未找到该指数的 K 线数据"}, ensure_ascii=False)
        
        return df.to_json(orient="records", force_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"get_index_daily_kline 失败：{e}")
        return json.dumps({"error": f"查询失败：{str(e)}"}, ensure_ascii=False)


@tool(args_schema=IndexWeightQuery)
def get_index_weight(
    index_code: Optional[str] = None,
    trade_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """获取指数成分股权重数据，查看指数中各股票的权重占比。"""
    try:
        # 验证指数代码（如果提供）
        if index_code:
            valid, err = validate_index_code(index_code)
            if not valid:
                return json.dumps({"error": f"指数代码验证失败：{err}"}, ensure_ascii=False)
        
        # 验证日期范围
        valid, err = validate_date_range(start_date, end_date)
        if not valid:
            return json.dumps({"error": f"日期验证失败：{err}"}, ensure_ascii=False)
        
        df = index_data.get_index_weight(
            index_code=index_code or "",
            trade_date=trade_date or "",
            start_date=start_date or "",
            end_date=end_date or "",
            limit=str(limit) if limit else "",
            offset=""
        )
        
        if df.empty:
            return json.dumps({"error": "未找到该指数的成分股权重数据"}, ensure_ascii=False)
        
        return df.to_json(orient="records", force_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"get_index_weight 失败：{e}")
        return json.dumps({"error": f"查询失败：{str(e)}"}, ensure_ascii=False)


@tool
def get_index_dailybasic(
    trade_date: Optional[str] = None,
    ts_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """获取指数每日基本面指标数据，包括总市值、流通市值、PE、PB 等指标。"""
    try:
        # 验证指数代码（如果提供）
        if ts_code:
            valid, err = validate_index_code(ts_code)
            if not valid:
                return json.dumps({"error": f"指数代码验证失败：{err}"}, ensure_ascii=False)
        
        # 验证日期范围
        valid, err = validate_date_range(start_date, end_date)
        if not valid:
            return json.dumps({"error": f"日期验证失败：{err}"}, ensure_ascii=False)
        
        df = index_data.get_index_dailybasic(
            trade_date=trade_date or "",
            ts_code=ts_code or "",
            start_date=start_date or "",
            end_date=end_date or "",
            limit=str(limit) if limit else "",
            offset=""
        )
        
        if df.empty:
            return json.dumps({"error": "未找到该指数的基本面数据"}, ensure_ascii=False)
        
        return df.to_json(orient="records", force_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"get_index_dailybasic 失败：{e}")
        return json.dumps({"error": f"查询失败：{str(e)}"}, ensure_ascii=False)
