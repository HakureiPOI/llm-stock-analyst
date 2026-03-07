"""
波动率预测特征工程 V2
预测目标：下一交易日的滚动窗口 Yang-Zhang 波动率
特征体系：基础日频特征 + OHLC区间特征 + GARCH特征

Yang-Zhang 波动率定义 (标准滚动窗口形式):
    σ²_YZ = σ²_o + k·σ²_c + (1-k)·σ²_RS
    
其中:
    - r_o = ln(O_t / C_{t-1})  隔夜收益率
    - r_c = ln(C_t / O_t)      日内收益率
    - RS = Rogers-Satchell 日内波动估计
    - k = 0.34 / (1.34 + (n+1)/(n-1))  最优权重
    - n 为滚动窗口长度
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import List, Optional, Tuple


class VolatilityTargets:
    """波动率目标变量计算"""

    @staticmethod
    def parkinson_volatility(high: np.ndarray, low: np.ndarray) -> float:
        """Parkinson 波动率 (单日)"""
        return np.sqrt((np.log(high / low) ** 2) / (4 * np.log(2)))

    @staticmethod
    def garman_klass_volatility(open_: np.ndarray, high: np.ndarray,
                                 low: np.ndarray, close: np.ndarray) -> float:
        """Garman-Klass 波动率 (单日)"""
        hl = np.log(high / low)
        co = np.log(close / open_)
        return np.sqrt(0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2)

    @staticmethod
    def rogers_satchell_volatility(open_: np.ndarray, high: np.ndarray,
                                    low: np.ndarray, close: np.ndarray) -> float:
        """Rogers-Satchell 波动率 (单日)"""
        u = np.log(high / open_)
        d = np.log(low / open_)
        c = np.log(close / open_)
        return np.sqrt(u * (u - c) + d * (d - c))

    @staticmethod
    def log_squared_return(close: np.ndarray, prev_close: np.ndarray,
                           epsilon: float = 1e-10) -> float:
        """对数平方收益率 (无OHLC时的替代目标)"""
        r = np.log(close / prev_close)
        return np.log(r ** 2 + epsilon)


class OHLCFeatures:
    """OHLC 区间特征"""

    @staticmethod
    def calculate_range_features(df: pd.DataFrame, yz_window: int = 20) -> pd.DataFrame:
        """
        计算所有区间型波动特征（含标准滚动窗口 Yang-Zhang 波动率）
        
        Args:
            df: 包含 OHLC 数据的 DataFrame
            yz_window: Yang-Zhang 滚动窗口长度，默认 20 日
            
        Returns:
            添加波动特征后的 DataFrame
        """
        df = df.copy()

        # 基础区间
        df['hl_range'] = np.log(df['high'] / df['low'])
        df['oc_range'] = np.abs(np.log(df['close'] / df['open']))
        df['co_gap'] = np.log(df['open'] / df['pre_close'])  # 隔夜跳空

        # Parkinson 波动率（单日）
        df['parkinson_vol'] = np.sqrt(
            (np.log(df['high'] / df['low']) ** 2) / (4 * np.log(2))
        )

        # Garman-Klass 波动率（单日）
        hl = np.log(df['high'] / df['open'])
        ll = np.log(df['low'] / df['open'])
        co = np.log(df['close'] / df['open'])
        df['garman_klass_vol'] = np.sqrt(
            0.5 * (hl - ll) ** 2 - (2 * np.log(2) - 1) * co ** 2
        )

        # Rogers-Satchell 波动率（单日）
        u = np.log(df['high'] / df['open'])
        d = np.log(df['low'] / df['open'])
        c = np.log(df['close'] / df['open'])
        rs = u * (u - c) + d * (d - c)
        df['rogers_satchell_vol'] = np.sqrt(np.maximum(rs, 0))

        # ========= 标准 Yang-Zhang（滚动窗口） =========
        # 隔夜收益率: r_o = ln(O_t / C_{t-1})
        r_o = np.log(df['open'] / df['pre_close'])
        
        # 日内收益率: r_c = ln(C_t / O_t)
        r_c = np.log(df['close'] / df['open'])

        n = yz_window
        if n <= 1:
            raise ValueError("yz_window must be > 1 for standard Yang-Zhang volatility.")

        # 最优权重 k
        k = 0.34 / (1.34 + (n + 1) / (n - 1))

        # 滚动窗口方差估计 (ddof=1 为样本方差)
        sigma_o2 = r_o.rolling(n).var(ddof=1)
        sigma_c2 = r_c.rolling(n).var(ddof=1)
        
        # Rogers-Satchell 波动率的滚动平均
        sigma_rs2 = rs.rolling(n).mean()

        # Yang-Zhang 方差估计
        yz_var = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2
        df['yang_zhang_vol'] = np.sqrt(np.maximum(yz_var, 0))

        return df

    @staticmethod
    def add_range_rolling_features(df: pd.DataFrame,
                                    windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """添加区间特征的滚动统计"""
        df = df.copy()

        range_cols = ['parkinson_vol', 'garman_klass_vol', 'rogers_satchell_vol',
                      'yang_zhang_vol', 'hl_range', 'oc_range', 'co_gap']

        for col in range_cols:
            if col not in df.columns:
                continue
            for w in windows:
                df[f'{col}_ma{w}'] = df[col].rolling(w).mean()
                df[f'{col}_std{w}'] = df[col].rolling(w).std()
                # Z-score
                df[f'{col}_zscore{w}'] = (df[col] - df[f'{col}_ma{w}']) / (df[f'{col}_std{w}'] + 1e-10)

        return df

    @staticmethod
    def add_range_lag_features(df: pd.DataFrame,
                               lags: List[int] = [1, 2, 5]) -> pd.DataFrame:
        """添加区间特征的滞后值"""
        df = df.copy()

        range_cols = ['parkinson_vol', 'garman_klass_vol', 'rogers_satchell_vol',
                      'yang_zhang_vol', 'hl_range', 'co_gap']

        for col in range_cols:
            if col not in df.columns:
                continue
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        return df


class DailyFeatures:
    """基础日频特征"""

    @staticmethod
    def calculate_return_features(df: pd.DataFrame) -> pd.DataFrame:
        """计算收益率相关特征"""
        df = df.copy()

        # 对数收益率
        df['log_ret'] = np.log(df['close'] / df['pre_close'])
        df['log_ret_sq'] = df['log_ret'] ** 2
        df['abs_ret'] = np.abs(df['log_ret'])

        return df

    @staticmethod
    def add_rolling_features(df: pd.DataFrame,
                             windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """添加滚动统计特征"""
        df = df.copy()

        for w in windows:
            # 滚动标准差
            df[f'ret_std_{w}'] = df['log_ret'].rolling(w).std()

            # 滚动绝对收益均值
            df[f'abs_ret_mean_{w}'] = df['abs_ret'].rolling(w).mean()

            # 下行波动率
            df[f'downside_vol_{w}'] = df['log_ret'].apply(
                lambda x: x if x < 0 else 0
            ).rolling(w).std()

            # 上行波动率
            df[f'upside_vol_{w}'] = df['log_ret'].apply(
                lambda x: x if x > 0 else 0
            ).rolling(w).std()

            # 滚动偏度
            df[f'ret_skew_{w}'] = df['log_ret'].rolling(w).apply(
                lambda x: stats.skew(x.dropna()) if len(x.dropna()) > 2 else 0
            )

            # 滚动峰度
            df[f'ret_kurt_{w}'] = df['log_ret'].rolling(w).apply(
                lambda x: stats.kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0
            )

            # 累计收益率
            df[f'ret_cum_{w}'] = df['log_ret'].rolling(w).sum()

        return df

    @staticmethod
    def add_volume_features(df: pd.DataFrame,
                            windows: List[int] = [5, 20]) -> pd.DataFrame:
        """添加成交量特征"""
        df = df.copy()

        df['log_vol'] = np.log(df['vol'].replace(0, np.nan))
        df['log_amt'] = np.log(df['amount'].replace(0, np.nan))

        for w in windows:
            df[f'log_vol_ma_{w}'] = df['log_vol'].rolling(w).mean()
            df[f'log_amt_ma_{w}'] = df['log_amt'].rolling(w).mean()
            df[f'vol_ratio_{w}'] = df['vol'] / df['vol'].rolling(w).mean()

        return df


class VolatilityFeatureEngineering:
    """波动率预测特征工程主类"""

    def __init__(self, yz_window: int = 20):
        """
        Args:
            yz_window: Yang-Zhang 滚动窗口长度，默认 20 日
        """
        self.yz_window = yz_window
        self.ohlc_features = OHLCFeatures()
        self.daily_features = DailyFeatures()

    def calculate_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算预测目标：下一交易日的滚动窗口 Yang-Zhang 波动率

        目标变量：y_t = log(σ_YZ,t+1 + ε)
        在 t 日收盘后可用 t 日及之前数据预测
        """
        df = df.copy()

        # 计算 t+1 的 Yang-Zhang 波动率
        df['target_vol'] = df['yang_zhang_vol'].shift(-1)

        # 对数变换 (使目标更稳定)
        df['target_vol_log'] = np.log(df['target_vol'] + 1e-10)

        return df

    def create_features(self, df: pd.DataFrame,
                        include_garch_features: bool = False) -> pd.DataFrame:
        """
        完整特征工程流程

        Args:
            df: 原始日线数据 (需包含 open, high, low, close, pre_close, vol, amount)
            include_garch_features: 是否包含GARCH特征 (需要单独计算)

        Returns:
            DataFrame: 特征工程后的数据
        """
        print(f"开始特征工程 (Yang-Zhang 窗口: {self.yz_window}日)...")

        # 排序
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 1. 基础收益率特征
        print("  - 计算收益率特征...")
        df = self.daily_features.calculate_return_features(df)

        # 2. 滚动统计特征
        print("  - 计算滚动统计特征...")
        df = self.daily_features.add_rolling_features(df)

        # 3. 成交量特征
        print("  - 计算成交量特征...")
        df = self.daily_features.add_volume_features(df)

        # 4. OHLC区间特征（含标准滚动窗口 Yang-Zhang）
        print(f"  - 计算OHLC区间特征 (YZ窗口={self.yz_window})...")
        df = self.ohlc_features.calculate_range_features(df, yz_window=self.yz_window)

        # 5. 区间特征滚动统计
        print("  - 计算区间特征滚动统计...")
        df = self.ohlc_features.add_range_rolling_features(df)

        # 6. 区间特征滞后值
        print("  - 计算区间特征滞后值...")
        df = self.ohlc_features.add_range_lag_features(df)

        # 7. 计算目标变量
        print("  - 计算目标变量 (Yang-Zhang t+1)...")
        df = self.calculate_target(df)

        # 统计
        feature_cols = self.get_feature_columns(df)
        print(f"特征工程完成: {len(df)} 条记录, {len(feature_cols)} 个特征")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """获取特征列名"""
        drop_cols = ['ts_code', 'trade_date', 'target_vol', 'target_vol_log',
                     'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
        return [col for col in df.columns if col not in drop_cols]


def create_volatility_features(input_file: str, output_file: str = None,
                                yz_window: int = 20) -> pd.DataFrame:
    """
    从数据文件创建特征

    Args:
        input_file: 输入数据文件路径
        output_file: 输出文件路径
        yz_window: Yang-Zhang 滚动窗口长度

    Returns:
        DataFrame: 包含特征的数据
    """
    fe = VolatilityFeatureEngineering(yz_window=yz_window)

    print(f"读取数据: {input_file}")
    df = pd.read_csv(input_file)
    print(f"原始数据: {len(df)} 条记录")

    # 特征工程
    df = fe.create_features(df)

    # 保存
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_features.csv"

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已保存到: {output_file}")

    return df


if __name__ == "__main__":
    ml_dir = Path(__file__).parent
    df = create_volatility_features(
        input_file=str(ml_dir / "dataset" / "index_daily_000001_SH_10years.csv"),
        output_file=str(ml_dir / "dataset" / "index_features.csv"),
        yz_window=20
    )
