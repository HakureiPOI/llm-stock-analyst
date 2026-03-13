"""技术分析和信号工具"""
from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import tool
from tsdata import stock_analyzer
from utils.validators import validate_stock_code, validate_index_code, validate_date_range
from utils.logger import setup_logger
import json

logger = setup_logger(__name__)


class TechnicalAnalysisQuery(BaseModel):
    ts_code: str = Field(..., description="股票或指数代码，如 '600000.SH' 或 '000001.SH'")
    start_date: Optional[str] = Field(None, description="开始日期，格式：YYYYMMDD，如 '20240101'")
    end_date: Optional[str] = Field(None, description="结束日期，格式：YYYYMMDD，如 '20241231'")


class StockAnalysisQuery(BaseModel):
    ts_code: str = Field(..., description="股票代码，如 '600000.SH'")
    start_date: Optional[str] = Field(None, description="开始日期，格式：YYYYMMDD")
    end_date: Optional[str] = Field(None, description="结束日期，格式：YYYYMMDD")


@tool(args_schema=TechnicalAnalysisQuery)
def analyze_stock_technical(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    对股票进行技术指标分析，计算包括：
    - 移动平均线：SMA5, SMA10, SMA20, SMA60
    - 指数移动平均：EMA12, EMA26
    - MACD 指标及信号
    - RSI 相对强弱指标（6, 12, 24 日）
    - 布林带（上、中、下轨）
    - KDJ 随机指标
    - ATR 平均真实波幅
    - OBV 能量潮
    - CCI 顺势指标
    - WR 威廉指标
    - 随机指标 Stochastic
    """
    try:
        # 验证股票代码
        valid, err = validate_stock_code(ts_code)
        if not valid:
            return json.dumps({"error": f"股票代码验证失败：{err}"}, ensure_ascii=False)
        
        # 验证日期范围
        valid, err = validate_date_range(start_date, end_date)
        if not valid:
            return json.dumps({"error": f"日期验证失败：{err}"}, ensure_ascii=False)
        
        df = stock_analyzer.get_stock_with_indicators(
            ts_code=ts_code,
            start_date=start_date or "",
            end_date=end_date or ""
        )

        if df.empty:
            return json.dumps(
                {"error": f"未找到股票 {ts_code} 的数据，请检查股票代码是否正确"},
                ensure_ascii=False
            )

        # 返回最近 5 天的数据
        recent_data = df.tail(5).to_dict(orient="records")

        result = {
            "stock_code": ts_code,
            "total_records": len(df),
            "recent_data": recent_data,
            "indicator_list": [
                "SMA_5", "SMA_10", "SMA_20", "SMA_60",
                "EMA_12", "EMA_26",
                "MACD", "MACD_Signal", "MACD_Hist",
                "RSI_6", "RSI_12", "RSI_24",
                "BB_Middle", "BB_Upper", "BB_Lower",
                "KDJ_K", "KDJ_D", "KDJ_J",
                "ATR", "Volume_SMA_5", "Volume_SMA_20",
                "OBV", "CCI", "WR", "Stoch_K", "Stoch_D"
            ]
        }

        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"analyze_stock_technical 失败：{e}")
        return json.dumps({"error": f"技术分析失败：{str(e)}"}, ensure_ascii=False)


@tool(args_schema=TechnicalAnalysisQuery)
def analyze_index_technical(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    对指数进行技术指标分析，计算与股票相同的技术指标（不含成交量相关指标）。
    包括移动平均、MACD、RSI、布林带、KDJ、ATR、CCI、WR 等。
    """
    try:
        # 验证指数代码
        valid, err = validate_index_code(ts_code)
        if not valid:
            return json.dumps({"error": f"指数代码验证失败：{err}"}, ensure_ascii=False)
        
        # 验证日期范围
        valid, err = validate_date_range(start_date, end_date)
        if not valid:
            return json.dumps({"error": f"日期验证失败：{err}"}, ensure_ascii=False)
        
        df = stock_analyzer.get_index_with_indicators(
            ts_code=ts_code,
            start_date=start_date or "",
            end_date=end_date or ""
        )

        if df.empty:
            return json.dumps(
                {"error": f"未找到指数 {ts_code} 的数据，请检查指数代码是否正确"},
                ensure_ascii=False
            )

        # 返回最近 5 天的数据
        recent_data = df.tail(5).to_dict(orient="records")

        result = {
            "index_code": ts_code,
            "total_records": len(df),
            "recent_data": recent_data,
            "indicator_list": [
                "SMA_5", "SMA_10", "SMA_20", "SMA_60",
                "EMA_12", "EMA_26",
                "MACD", "MACD_Signal", "MACD_Hist",
                "RSI_6", "RSI_12", "RSI_24",
                "BB_Middle", "BB_Upper", "BB_Lower",
                "KDJ_K", "KDJ_D", "KDJ_J",
                "ATR", "CCI", "WR", "Stoch_K", "Stoch_D"
            ]
        }

        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"analyze_index_technical 失败：{e}")
        return json.dumps({"error": f"技术分析失败：{str(e)}"}, ensure_ascii=False)


@tool(args_schema=StockAnalysisQuery)
def analyze_stock_signals(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    基于技术指标生成股票买卖信号和投资建议。

    分析内容包括：
    - MA 趋势信号（短期与长期均线的关系）
    - MACD 金叉死叉信号
    - RSI 超买超卖信号
    - KDJ 买卖信号
    - 布林带突破信号
    - CCI 趋势信号
    - 综合建议
    """
    try:
        # 验证股票代码
        valid, err = validate_stock_code(ts_code)
        if not valid:
            return json.dumps({"error": f"股票代码验证失败：{err}"}, ensure_ascii=False)
        
        # 验证日期范围
        valid, err = validate_date_range(start_date, end_date)
        if not valid:
            return json.dumps({"error": f"日期验证失败：{err}"}, ensure_ascii=False)
        
        df = stock_analyzer.get_stock_with_indicators(
            ts_code=ts_code,
            start_date=start_date or "",
            end_date=end_date or ""
        )

        if df.empty or len(df) < 20:
            return json.dumps(
                {"error": f"数据不足，无法生成信号。股票 {ts_code} 至少需要 20 天数据"},
                ensure_ascii=False
            )

        df = stock_analyzer.analyze_signal(df)
        latest = df.iloc[-1]

        # 构建信号分析结果
        signals = {
            "stock_code": ts_code,
            "current_price": float(latest['close']),
            "trade_date": str(latest['trade_date']),
            "technical_indicators": {
                "ma_trend": {
                    "SMA5": float(latest['SMA_5']),
                    "SMA20": float(latest['SMA_20']),
                    "signal": "上涨趋势" if latest['SMA_5'] > latest['SMA_20'] else "下跌趋势"
                },
                "macd": {
                    "MACD": float(latest['MACD']),
                    "Signal": float(latest['MACD_Signal']),
                    "Histogram": float(latest['MACD_Hist']),
                    "signal": "金叉" if latest['MACD'] > latest['MACD_Signal'] else "死叉"
                },
                "rsi": {
                    "RSI12": float(latest['RSI_12']),
                    "signal": "超买" if latest['RSI_12'] > 70 else ("超卖" if latest['RSI_12'] < 30 else "中性")
                },
                "kdj": {
                    "K": float(latest['KDJ_K']),
                    "D": float(latest['KDJ_D']),
                    "J": float(latest['KDJ_J']),
                    "signal": "买入" if latest['KDJ_K'] > latest['KDJ_D'] and latest['KDJ_K'] < 20 else ("卖出" if latest['KDJ_K'] < latest['KDJ_D'] and latest['KDJ_K'] > 80 else "观望")
                },
                "bollinger": {
                    "Upper": float(latest['BB_Upper']),
                    "Middle": float(latest['BB_Middle']),
                    "Lower": float(latest['BB_Lower']),
                    "signal": "触及下轨" if latest['close'] < latest['BB_Lower'] else ("触及上轨" if latest['close'] > latest['BB_Upper'] else "在中轨附近")
                }
            },
            "all_signals": str(latest['signals']),
            "overall_advice": generate_advice(latest)
        }

        return json.dumps(signals, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"analyze_stock_signals 失败：{e}")
        return json.dumps({"error": f"信号分析失败：{str(e)}"}, ensure_ascii=False)


def generate_advice(latest_data) -> str:
    """生成综合投资建议"""
    advice_parts = []

    # MA 趋势
    if latest_data['SMA_5'] > latest_data['SMA_20']:
        advice_parts.append("短期均线向上，短期偏强")
    else:
        advice_parts.append("短期均线向下，短期偏弱")

    # MACD
    if latest_data['MACD'] > latest_data['MACD_Signal']:
        advice_parts.append("MACD 金叉，可能上涨")
    else:
        advice_parts.append("MACD 死叉，可能下跌")

    # RSI
    if latest_data['RSI_12'] > 70:
        advice_parts.append("RSI 超买，注意回调风险")
    elif latest_data['RSI_12'] < 30:
        advice_parts.append("RSI 超卖，可能有反弹机会")

    # 综合判断
    buy_signals = 0
    sell_signals = 0

    if latest_data['SMA_5'] > latest_data['SMA_20']:
        buy_signals += 1
    else:
        sell_signals += 1

    if latest_data['MACD'] > latest_data['MACD_Signal']:
        buy_signals += 1
    else:
        sell_signals += 1

    if latest_data['RSI_12'] < 30:
        buy_signals += 1
    elif latest_data['RSI_12'] > 70:
        sell_signals += 1

    if buy_signals >= 2:
        advice_parts.insert(0, "【综合建议】多个指标看涨，可考虑逢低买入")
    elif sell_signals >= 2:
        advice_parts.insert(0, "【综合建议】多个指标看跌，建议谨慎观望或减仓")
    else:
        advice_parts.insert(0, "【综合建议】指标分化，建议继续观察")

    return " | ".join(advice_parts)


@tool
def get_indicator_explanation() -> str:
    """
    获取各个技术指标的详细解释和使用方法。
    """
    explanation = {
        "移动平均线 (SMA/EMA)": "平滑价格波动，用于判断趋势。短期上穿长期为金叉，下穿为死叉。",
        "MACD": "指数平滑异同移动平均线，用于判断趋势和动能。柱状图上穿零轴为买入信号。",
        "RSI": "相对强弱指数，衡量买卖力量。>70 超买（可能回调），<30 超卖（可能反弹）。",
        "布林带 (BB)": "基于标准差的通道。价格触及上轨可能回调，触及下轨可能反弹。",
        "KDJ": "随机指标，用于捕捉超买超卖。K 线上穿 D 线为买入信号，J 值反映超买超卖程度。",
        "ATR": "平均真实波幅，衡量价格波动性，用于设置止损。",
        "OBV": "能量潮，根据成交量变化反映资金流向。",
        "CCI": "顺势指标，用于判断价格偏离常态的程度。>100 超买，<-100 超卖。",
        "WR (威廉指标)": "与 RSI 类似的震荡指标。-80~-100 超买，0~-20 超卖。",
        "Stochastic": "随机指标，KD 线交叉提供买卖信号，与 KDJ 类似。"
    }

    result = []
    for indicator, desc in explanation.items():
        result.append(f"**{indicator}**\n{desc}\n")

    return "\n".join(result)
