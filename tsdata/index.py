from .client import pro

class IndexData:
    """指数数据获取类"""
    
    def get_index_basic(self, ts_code="", market="", publisher="", category="", name="", limit="", offset=""):
        """
        获取指数基本信息
        
        Args:
            ts_code: 指数代码，支持逗号分隔的多个代码
            market: 市场
            publisher: 发布商
            category: 类别
            name: 指数名称
            limit: 限制返回数量
            offset: 偏移量
            
        Returns:
            DataFrame: 指数基本信息数据
        """
        return pro.index_basic(**{
            "ts_code": ts_code,
            "market": market,
            "publisher": publisher,
            "category": category,
            "name": name,
            "limit": limit,
            "offset": offset
        }, fields=[
            "ts_code",
            "name",
            "market",
            "publisher",
            "category",
            "base_date",
            "base_point",
            "list_date",
            "fullname",
            "index_type",
            "weight_rule",
            "desc",
            "exp_date"
        ])
    
    def get_index_daily(self, ts_code, trade_date="", start_date="", end_date="", limit="", offset=""):
        """
        获取指数日K线数据
        
        Args:
            ts_code: 指数代码，不支持逗号分隔的多个代码
            trade_date: 交易日期，YYYYMMDD格式
            start_date: 开始日期，YYYYMMDD格式
            end_date: 结束日期，YYYYMMDD格式
            limit: 限制返回数量
            offset: 偏移量
            
        Returns:
            DataFrame: 指数日K线数据
        """
        return pro.index_daily(**{
            "ts_code": ts_code,
            "trade_date": trade_date,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "offset": offset
        }, fields=[
            "ts_code",
            "trade_date",
            "close",
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])
    
    def get_index_weight(self, index_code="", trade_date="", start_date="", end_date="", ts_code="", limit="", offset=""):
        """
        获取指数成分股权重数据
        
        Args:
            index_code: 指数代码
            trade_date: 交易日期，YYYYMMDD格式
            start_date: 开始日期，YYYYMMDD格式
            end_date: 结束日期，YYYYMMDD格式
            ts_code: 股票代码
            limit: 限制返回数量
            offset: 偏移量
            
        Returns:
            DataFrame: 指数成分股权重数据
        """
        return pro.index_weight(**{
            "index_code": index_code,
            "trade_date": trade_date,
            "start_date": start_date,
            "end_date": end_date,
            "ts_code": ts_code,
            "limit": limit,
            "offset": offset
        }, fields=[
            "index_code",
            "con_code",
            "trade_date",
            "weight"
        ])
    
    def get_index_dailybasic(self, trade_date="", ts_code="", start_date="", end_date="", limit="", offset=""):
        """
        获取指数每日基本面指标数据
        
        Args:
            trade_date: 交易日期，YYYYMMDD格式
            ts_code: 指数代码
            start_date: 开始日期，YYYYMMDD格式
            end_date: 结束日期，YYYYMMDD格式
            limit: 限制返回数量
            offset: 偏移量
            
        Returns:
            DataFrame: 指数每日基本面指标数据
        """
        return pro.index_dailybasic(**{
            "trade_date": trade_date,
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "offset": offset
        }, fields=[
            "ts_code",
            "trade_date",
            "total_mv",
            "float_mv",
            "total_share",
            "float_share",
            "free_share",
            "turnover_rate",
            "turnover_rate_f",
            "pe",
            "pe_ttm",
            "pb"
        ])


index_data = IndexData()