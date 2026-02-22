import pandas as pd
import numpy as np
from pathlib import Path


class FeatureEngineering:
    """特征工程类 - 参考volatility.py方案"""

    def calculate_future_volatility(self, df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
        """
        计算未来波动率: 未来N日对数收益率的标准差

        Args:
            df: 输入数据
            window_size: 波动率窗口天数,默认5天

        Returns:
            DataFrame: 添加了log_return和future_volatility的数据
        """
        df = df.sort_values(by=['ts_code', 'trade_date']).copy()

        # 计算日对数收益率
        df['log_return'] = df.groupby('ts_code')['close'].transform(
            lambda x: np.log(x / x.shift(1))
        )

        # 计算未来波动率: 未来window_size天对数收益率的标准差
        df['future_volatility'] = df.groupby('ts_code')['log_return'].transform(
            lambda x: x.shift(-1).rolling(window=window_size).std()
        )

        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加特征: 滞后特征、移动平均、技术指标、日期特征

        Args:
            df: 已计算log_return和future_volatility的数据

        Returns:
            DataFrame: 添加了所有特征的数据
        """
        df = df.copy()

        # 1. 滞后特征: close, vol, log_return
        print("生成滞后期特征...")
        for col in ['close', 'vol', 'log_return']:
            for lag in [1, 3, 5]:
                df[f'{col}_lag{lag}'] = df.groupby('ts_code')[col].shift(lag)

        # 2. 移动平均: close价格
        print("生成移动平均特征...")
        for window in [5, 10, 20]:
            df[f'MA_close_{window}'] = df.groupby('ts_code')['close'].transform(
                lambda x: x.rolling(window=window).mean()
            )

        # 3. RSI指标 (14日)
        print("生成RSI指标...")
        def calculate_rsi(series, window=14):
            diff = series.diff(1)
            gain = diff.mask(diff < 0, 0)
            loss = diff.mask(diff > 0, 0).abs()
            avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
            avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        df['RSI_14'] = df.groupby('ts_code')['close'].transform(
            lambda x: calculate_rsi(x, window=14)
        )

        # 4. MACD指标
        print("生成MACD指标...")
        # 计算EMA
        df['EMA_fast'] = df.groupby('ts_code')['close'].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
        df['EMA_slow'] = df.groupby('ts_code')['close'].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )

        # MACD线
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']

        # 信号线
        df['Signal_Line'] = df.groupby('ts_code')['MACD'].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )

        # 柱状图
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

        # 删除中间列
        df = df.drop(columns=['EMA_fast', 'EMA_slow'])

        # 5. 日期特征
        print("生成日期特征...")
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df['day_of_week'] = df['trade_date'].dt.dayofweek
        df['month'] = df['trade_date'].dt.month

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理数据: 删除含NaN的行

        Args:
            df: 特征工程后的数据

        Returns:
            DataFrame: 清理后的数据
        """
        original_len = len(df)
        df = df.dropna()
        print(f"清理数据: {original_len} -> {len(df)} 条记录")
        return df

    def create_features(self, input_file: str, output_file: str = None,
                     window_size: int = 5) -> pd.DataFrame:
        """
        完整特征工程流程

        Args:
            input_file: 输入数据文件路径
            output_file: 输出文件路径
            window_size: 波动率计算窗口,默认5天

        Returns:
            DataFrame: 包含特征和目标的完整数据集
        """
        print(f"读取数据: {input_file}")
        df = pd.read_csv(input_file)
        print(f"原始数据: {len(df)} 条记录, {df['ts_code'].nunique()} 只股票")

        # 1. 计算未来波动率 (目标变量)
        print(f"\n步骤1: 计算未来{window_size}日波动率...")
        df = self.calculate_future_volatility(df, window_size)

        # 2. 添加特征
        print("\n步骤2: 生成特征...")
        df = self.add_features(df)

        # 3. 清理数据
        print("\n步骤3: 清理数据...")
        df = self.clean_data(df)

        # 4. 保存
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_features.csv"

        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n已保存到: {output_file}")
        print(f"\n最终数据: {len(df)} 条记录, {df['ts_code'].nunique()} 只股票")
        print(f"特征列数: {len(df.columns)}")

        return df


if __name__ == "__main__":
    fe = FeatureEngineering()
    dataset = fe.create_features(
        input_file="/home/hakurei/llm-stock-analyst/ml/dataset/kline_dataset_000016_SH_5years.csv",
        output_file="/home/hakurei/llm-stock-analyst/ml/dataset/kline_dataset_features.csv",
        window_size=5
    )
