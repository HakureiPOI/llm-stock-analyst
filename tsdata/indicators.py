"""技术指标计算模块"""
import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats

from .stock import stock_data
from .index import index_data
from utils.logger import setup_logger
from utils.validators import validate_stock_code, validate_index_code

logger = setup_logger(__name__)


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
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """MACD 指标"""
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
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: int = 2
    ) -> pd.DataFrame:
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
    def kdj(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        n: int = 9,
        m1: int = 3,
        m2: int = 3
    ) -> pd.DataFrame:
        """KDJ 指标"""
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
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
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
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """顺势指标"""
        tp = (high + low + close) / 3
        ma = tp.rolling(window=period).mean()
        md = tp.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=False
        )

        cci = (tp - ma) / (0.015 * md)
        return cci

    @staticmethod
    def wr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """威廉指标"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        wr = (highest_high - close) / (highest_high - lowest_low) * -100
        return wr

    @staticmethod
    def stoch(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """随机指标"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = (close - lowest_low) / (highest_high - lowest_low) * 100
        d_percent = k_percent.rolling(window=d_period).mean()

        return pd.DataFrame({
            'K': k_percent,
            'D': d_percent
        })


class StockAnalyzer:
    """股票技术分析类"""

    def __init__(self):
        self.indicators = TechnicalIndicators()

    def get_stock_with_indicators(
        self,
        ts_code: str,
        start_date: str = "",
        end_date: str = ""
    ) -> pd.DataFrame:
        """
        获取股票数据并计算技术指标

        Args:
            ts_code: 股票代码
            start_date: 开始日期，YYYYMMDD 格式
            end_date: 结束日期，YYYYMMDD 格式

        Returns:
            DataFrame: 包含原始数据和技术指标的 DataFrame
        """
        # 验证股票代码
        valid, err = validate_stock_code(ts_code)
        if not valid:
            logger.error(f"股票代码验证失败：{err}")
            return pd.DataFrame()
        
        logger.debug(f"计算股票技术指标：ts_code={ts_code}")
        
        try:
            df = stock_data.get_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )

            if df.empty:
                logger.warning(f"未获取到股票数据：ts_code={ts_code}")
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

            logger.debug(f"技术指标计算完成：{len(df)}条记录")
            return df
            
        except Exception as e:
            logger.error(f"计算股票技术指标失败：{e}")
            return pd.DataFrame()

    def get_index_with_indicators(
        self,
        ts_code: str,
        start_date: str = "",
        end_date: str = ""
    ) -> pd.DataFrame:
        """
        获取指数数据并计算技术指标

        Args:
            ts_code: 指数代码
            start_date: 开始日期，YYYYMMDD 格式
            end_date: 结束日期，YYYYMMDD 格式

        Returns:
            DataFrame: 包含原始数据和技术指标的 DataFrame
        """
        # 验证指数代码
        valid, err = validate_index_code(ts_code)
        if not valid:
            logger.error(f"指数代码验证失败：{err}")
            return pd.DataFrame()
        
        logger.debug(f"计算指数技术指标：ts_code={ts_code}")
        
        try:
            df = index_data.get_index_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )

            if df.empty:
                logger.warning(f"未获取到指数数据：ts_code={ts_code}")
                return df

            df = df.sort_values('trade_date').reset_index(drop=True)
            close = df['close']
            high = df['high']
            low = df['low']

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

            logger.debug(f"指数技术指标计算完成：{len(df)}条记录")
            return df
            
        except Exception as e:
            logger.error(f"计算指数技术指标失败：{e}")
            return pd.DataFrame()

    def analyze_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基于技术指标生成交易信号

        Args:
            df: 包含技术指标的 DataFrame

        Returns:
            DataFrame: 添加了信号列的 DataFrame
        """
        if df.empty or len(df) < 20:
            logger.warning("数据不足，无法生成信号")
            return df

        latest = df.iloc[-1]

        signals = []

        # MA 信号
        if latest['SMA_5'] > latest['SMA_20']:
            signals.append("MA 多头")
        elif latest['SMA_5'] < latest['SMA_20']:
            signals.append("MA 空头")

        # MACD 信号
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > 0:
            signals.append("MACD 金叉")
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < 0:
            signals.append("MACD 死叉")

        # RSI 信号
        if latest['RSI_12'] < 30:
            signals.append("RSI 超卖")
        elif latest['RSI_12'] > 70:
            signals.append("RSI 超买")

        # KDJ 信号
        if latest['KDJ_K'] > latest['KDJ_D'] and latest['KDJ_K'] < 20:
            signals.append("KDJ 低位金叉")
        elif latest['KDJ_K'] < latest['KDJ_D'] and latest['KDJ_K'] > 80:
            signals.append("KDJ 高位死叉")

        # 布林带信号
        if latest['close'] < latest['BB_Lower']:
            signals.append("触及布林下轨")
        elif latest['close'] > latest['BB_Upper']:
            signals.append("触及布林上轨")

        # CCI 信号
        if latest['CCI'] < -100:
            signals.append("CCI 超卖")
        elif latest['CCI'] > 100:
            signals.append("CCI 超买")

        df['signals'] = ', '.join(signals) if signals else "无明显信号"

        return df


# 创建全局实例
stock_analyzer = StockAnalyzer()
