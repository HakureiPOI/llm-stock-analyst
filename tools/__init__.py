# 股票分析工具包
# 导入所有工具供 LangChain 使用

from .stock_tools import (
    get_stock_basic_info,
    get_stock_daily_kline,
    get_stock_financial_data,
    get_stock_income,
    get_stock_balance_sheet,
    get_stock_cashflow,
    get_stock_fina_indicator,
    get_adj_factor,
)

from .index_tools import (
    get_index_basic_info,
    get_index_daily_kline,
    get_index_weight,
    get_index_dailybasic,
)

from .analysis_tools import (
    analyze_stock_technical,
    analyze_index_technical,
    analyze_stock_signals,
    get_indicator_explanation,
)

from .common_tools import (
    get_current_time,
    get_stock_market_status,
)

from .volatility_tools import (
    predict_index_volatility,
    compare_index_volatility,
    get_market_volatility_summary,
)

# 兼容旧名称
predict_index_risk = predict_index_volatility
compare_index_risk = compare_index_volatility
get_market_risk_summary = get_market_volatility_summary

from .websearch_tools import web_search

__all__ = [
    # 股票工具
    "get_stock_basic_info",
    "get_stock_daily_kline",
    "get_stock_financial_data",
    "get_stock_income",
    "get_stock_balance_sheet",
    "get_stock_cashflow",
    "get_stock_fina_indicator",
    "get_adj_factor",
    # 指数工具
    "get_index_basic_info",
    "get_index_daily_kline",
    "get_index_weight",
    "get_index_dailybasic",
    # 分析工具
    "analyze_stock_technical",
    "analyze_index_technical",
    "analyze_stock_signals",
    "get_indicator_explanation",
    # 通用工具
    "get_current_time",
    "get_stock_market_status",
    # 指数风险度预测工具
    "predict_index_risk",
    "compare_index_risk",
    "get_market_risk_summary",
    # 联网搜索工具
    "web_search",
]
