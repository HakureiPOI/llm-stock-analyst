import pandas as pd
import numpy as np
from .stock import stock_data


class TechnicalIndicators:
    """技术指标计算类"""

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """简单移动平均"""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """指数移动平均"""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """MACD指标"""
        ema_fast = TechnicalIndicators.ema(data, fast_period)
        ema_slow = TechnicalIndicators.ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """相对强弱指数"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """布林带"""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return pd.DataFrame({
            'Middle': sma,
            'Upper': upper_band,
            'Lower': lower_band
        })

    @staticmethod
    def kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """KDJ指标"""
        lowest_low = low.rolling(window=n).min()
        highest_high = high.rolling(window=n).max()

        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100

        k = rsv.ewm(com=m1 - 1, adjust=False).mean()
        d = k.ewm(com=m2 - 1, adjust=False).mean()
        j = 3 * k - 2 * d

        return pd.DataFrame({
            'K': k,
            'D': d,
            'J': j
        })

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """平均真实波幅"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 5) -> pd.Series:
        """成交量移动平均"""
        return volume.rolling(window=period).mean()

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """能量潮指标"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """顺势指标"""
        tp = (high + low + close) / 3
        ma = tp.rolling(window=period).mean()
        md = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

        cci = (tp - ma) / (0.015 * md)
        return cci

    @staticmethod
    def wr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """威廉指标"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        wr = (highest_high - close) / (highest_high - lowest_low) * -100
        return wr

    @staticmethod
    def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """随机指标"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = (close - lowest_low) / (highest_high - lowest_low) * 100
        d_percent = k_percent.rolling(window=d_period).mean()

        return pd.DataFrame({
            'K': k_percent,
            'D': d_percent
        })

    @staticmethod
    def psar(high: pd.Series, low: pd.Series, af: float = 0.02, max_af: float = 0.2) -> pd.Series:
        """抛物线转向 - 简化实现"""
        # 使用numpy纯数组处理
        h_arr = high.to_numpy()
        l_arr = low.to_numpy()
        n = len(h_arr)

        psar_arr = np.full(n, np.nan)
        if n == 0:
            return pd.Series(psar_arr, index=high.index)

        psar_arr[0] = l_arr[0]
        if n > 1:
            psar_arr[1] = l_arr[1]

        # 简单追踪逻辑
        for i in range(2, n):
            window_start = max(0, i - 5)
            recent_low = np.nanmin(l_arr[window_start:i+1])
            recent_high = np.nanmax(h_arr[window_start:i+1])

            prev_close = (h_arr[i-1] + l_arr[i-1]) / 2

            if prev_close < psar_arr[i-1]:
                psar_arr[i] = recent_low * (1 + af)
            else:
                psar_arr[i] = recent_high * (1 - af)

        return pd.Series(psar_arr, index=high.index)


class StockAnalyzer:
    """股票技术分析类"""

    def __init__(self):
        self.indicators = TechnicalIndicators()

    def get_stock_with_indicators(self, ts_code: str, start_date: str = "", end_date: str = "") -> pd.DataFrame:
        """
        获取股票数据并计算技术指标

        Args:
            ts_code: 股票代码
            start_date: 开始日期，YYYYMMDD格式
            end_date: 结束日期，YYYYMMDD格式

        Returns:
            DataFrame: 包含原始数据和技术指标的DataFrame
        """
        df = stock_data.get_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        if df.empty:
            return df

        df = df.sort_values('trade_date').reset_index(drop=True)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['vol']

        # 计算各类指标
        df['SMA_5'] = self.indicators.sma(close, 5)
        df['SMA_10'] = self.indicators.sma(close, 10)
        df['SMA_20'] = self.indicators.sma(close, 20)
        df['SMA_60'] = self.indicators.sma(close, 60)

        df['EMA_12'] = self.indicators.ema(close, 12)
        df['EMA_26'] = self.indicators.ema(close, 26)

        macd_df = self.indicators.macd(close)
        df['MACD'] = macd_df['MACD']
        df['MACD_Signal'] = macd_df['Signal']
        df['MACD_Hist'] = macd_df['Histogram']

        df['RSI_6'] = self.indicators.rsi(close, 6)
        df['RSI_12'] = self.indicators.rsi(close, 12)
        df['RSI_24'] = self.indicators.rsi(close, 24)

        bb_df = self.indicators.bollinger_bands(close)
        df['BB_Middle'] = bb_df['Middle']
        df['BB_Upper'] = bb_df['Upper']
        df['BB_Lower'] = bb_df['Lower']

        kdj_df = self.indicators.kdj(high, low, close)
        df['KDJ_K'] = kdj_df['K']
        df['KDJ_D'] = kdj_df['D']
        df['KDJ_J'] = kdj_df['J']

        df['ATR'] = self.indicators.atr(high, low, close)

        df['Volume_SMA_5'] = self.indicators.volume_sma(volume, 5)
        df['Volume_SMA_20'] = self.indicators.volume_sma(volume, 20)

        df['OBV'] = self.indicators.obv(close, volume)

        df['CCI'] = self.indicators.cci(high, low, close)

        df['WR'] = self.indicators.wr(high, low, close)

        stoch_df = self.indicators.stoch(high, low, close)
        df['Stoch_K'] = stoch_df['K']
        df['Stoch_D'] = stoch_df['D']

        # df['PSAR'] = self.indicators.psar(high, low, close)  # 暂时禁用PSAR

        return df

    def get_index_with_indicators(self, ts_code: str, start_date: str = "", end_date: str = "") -> pd.DataFrame:
        """
        获取指数数据并计算技术指标

        Args:
            ts_code: 指数代码
            start_date: 开始日期，YYYYMMDD格式
            end_date: 结束日期，YYYYMMDD格式

        Returns:
            DataFrame: 包含原始数据和技术指标的DataFrame
        """
        from .index import index_data

        df = index_data.get_index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        if df.empty:
            return df

        df = df.sort_values('trade_date').reset_index(drop=True)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['vol']

        # 指数不需要成交量指标，计算价格相关指标
        df['SMA_5'] = self.indicators.sma(close, 5)
        df['SMA_10'] = self.indicators.sma(close, 10)
        df['SMA_20'] = self.indicators.sma(close, 20)
        df['SMA_60'] = self.indicators.sma(close, 60)

        df['EMA_12'] = self.indicators.ema(close, 12)
        df['EMA_26'] = self.indicators.ema(close, 26)

        macd_df = self.indicators.macd(close)
        df['MACD'] = macd_df['MACD']
        df['MACD_Signal'] = macd_df['Signal']
        df['MACD_Hist'] = macd_df['Histogram']

        df['RSI_6'] = self.indicators.rsi(close, 6)
        df['RSI_12'] = self.indicators.rsi(close, 12)
        df['RSI_24'] = self.indicators.rsi(close, 24)

        bb_df = self.indicators.bollinger_bands(close)
        df['BB_Middle'] = bb_df['Middle']
        df['BB_Upper'] = bb_df['Upper']
        df['BB_Lower'] = bb_df['Lower']

        kdj_df = self.indicators.kdj(high, low, close)
        df['KDJ_K'] = kdj_df['K']
        df['KDJ_D'] = kdj_df['D']
        df['KDJ_J'] = kdj_df['J']

        df['ATR'] = self.indicators.atr(high, low, close)

        df['CCI'] = self.indicators.cci(high, low, close)

        df['WR'] = self.indicators.wr(high, low, close)

        stoch_df = self.indicators.stoch(high, low, close)
        df['Stoch_K'] = stoch_df['K']
        df['Stoch_D'] = stoch_df['D']

        # df['PSAR'] = self.indicators.psar(high, low, close)  # 暂时禁用PSAR

        return df

    def analyze_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基于技术指标生成交易信号

        Returns:
            DataFrame: 添加了信号列的DataFrame
        """
        if df.empty or len(df) < 20:
            return df

        latest = df.iloc[-1]

        signals = []

        # MA信号
        if latest['SMA_5'] > latest['SMA_20']:
            signals.append("MA多头")
        elif latest['SMA_5'] < latest['SMA_20']:
            signals.append("MA空头")

        # MACD信号
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > 0:
            signals.append("MACD金叉")
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < 0:
            signals.append("MACD死叉")

        # RSI信号
        if latest['RSI_12'] < 30:
            signals.append("RSI超卖")
        elif latest['RSI_12'] > 70:
            signals.append("RSI超买")

        # KDJ信号
        if latest['KDJ_K'] > latest['KDJ_D'] and latest['KDJ_K'] < 20:
            signals.append("KDJ低位金叉")
        elif latest['KDJ_K'] < latest['KDJ_D'] and latest['KDJ_K'] > 80:
            signals.append("KDJ高位死叉")

        # 布林带信号
        if latest['close'] < latest['BB_Lower']:
            signals.append("触及布林下轨")
        elif latest['close'] > latest['BB_Upper']:
            signals.append("触及布林上轨")

        # CCI信号
        if latest['CCI'] < -100:
            signals.append("CCI超卖")
        elif latest['CCI'] > 100:
            signals.append("CCI超买")

        df['signals'] = ', '.join(signals) if signals else "无明显信号"

        return df


# 创建全局实例
stock_analyzer = StockAnalyzer()
