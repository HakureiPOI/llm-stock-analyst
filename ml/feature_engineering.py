"""指数风险度特征工程 - 基于已实现波动率预测"""
import pandas as pd
import numpy as np
from pathlib import Path


class IndexRiskFeatureEngineering:
    """指数风险度特征工程类"""

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算收益率相关特征

        Args:
            df: 指数日线数据 (需包含 close, pre_close, high, low)

        Returns:
            DataFrame: 添加了收益率特征的数据
        """
        df = df.copy()

        # 对数收益率
        df['log_ret'] = np.log(df['close'] / df['pre_close'])

        # 收益率平方 (用于波动率计算)
        df['r2'] = df['log_ret'] ** 2

        # 绝对收益率
        df['abs_ret'] = np.abs(df['log_ret'])

        # 振幅与开盘跳空
        df['hl_spread'] = np.log(df['high'] / df['low'])
        df['oc_ret'] = np.log(df['close'] / df['open'])
        df['gap_ret'] = np.log(df['open'] / df['pre_close'])

        return df

    def calculate_rolling_distribution_features(self, df: pd.DataFrame, windows: list = [5, 20, 60]) -> pd.DataFrame:
        """计算滚动分布特征（偏度/峰度/分位差）。"""
        df = df.copy()

        for w in windows:
            rolling_log_ret = df['log_ret'].rolling(w)
            df[f'log_ret_std_{w}'] = rolling_log_ret.std()
            df[f'log_ret_skew_{w}'] = rolling_log_ret.skew()
            df[f'log_ret_kurt_{w}'] = rolling_log_ret.kurt()
            q75 = rolling_log_ret.quantile(0.75)
            q25 = rolling_log_ret.quantile(0.25)
            df[f'log_ret_iqr_{w}'] = q75 - q25

        return df

    def calculate_future_rv(self, df: pd.DataFrame, horizons: list = [5, 20]) -> pd.DataFrame:
        """
        计算未来已实现波动率 (目标变量)

        Args:
            df: 包含 r2 列的数据
            horizons: 预测窗口列表

        Returns:
            DataFrame: 添加了目标变量的数据
        """
        df = df.copy()

        def future_rv(r2, h):
            """未来 h 日已实现波动率"""
            return np.sqrt(r2.shift(-1).rolling(h).sum()).shift(-(h-1))

        for h in horizons:
            df[f'rv_{h}_fut'] = future_rv(df['r2'], h)

        return df

    def calculate_past_rv(self, df: pd.DataFrame, windows: list = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        计算过去已实现波动率 (特征)

        Args:
            df: 包含 r2 列的数据
            windows: 计算窗口列表

        Returns:
            DataFrame: 添加了过去波动率特征的数据
        """
        df = df.copy()

        for w in windows:
            df[f'rv_past_{w}'] = np.sqrt(df['r2'].rolling(w).sum())

        return df

    def calculate_parkinson_var(self, df: pd.DataFrame, windows: list = [5, 20]) -> pd.DataFrame:
        """
        计算 Parkinson 方差代理 (基于高低价)

        Args:
            df: 包含 high, low 列的数据
            windows: 计算窗口列表

        Returns:
            DataFrame: 添加了 Parkinson 方差特征的数据
        """
        df = df.copy()

        # Parkinson 方差代理
        df['log_hl'] = np.log(df['high'] / df['low'])
        df['pk_var'] = (df['log_hl'] ** 2) / (4 * np.log(2))

        # 累积 Parkinson 波动率
        for w in windows:
            df[f'pk_rv_past_{w}'] = np.sqrt(df['pk_var'].rolling(w).sum())

        return df

    def calculate_volume_features(self, df: pd.DataFrame, windows: list = [5, 20]) -> pd.DataFrame:
        """
        计算成交量相关特征

        Args:
            df: 包含 vol, amount 列的数据
            windows: 计算窗口列表

        Returns:
            DataFrame: 添加了成交量特征的数据
        """
        df = df.copy()

        # 对数成交量和成交额
        df['log_vol'] = np.log(df['vol'].replace(0, np.nan))
        df['log_amt'] = np.log(df['amount'].replace(0, np.nan))

        # 移动平均
        for w in windows:
            df[f'log_vol_ma_{w}'] = df['log_vol'].rolling(w).mean()
            df[f'log_amt_ma_{w}'] = df['log_amt'].rolling(w).mean()

        return df

    def calculate_rv_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算波动率比率特征

        Args:
            df: 包含 rv_past_* 列的数据

        Returns:
            DataFrame: 添加了波动率比率特征的数据
        """
        df = df.copy()

        # 波动率比率
        df['rv_ratio_5_20'] = df['rv_past_5'] / df['rv_past_20']
        df['rv_ratio_20_60'] = df['rv_past_20'] / df['rv_past_60']

        # 波动率变化率
        df['rv5_chg'] = df['rv_past_5'].pct_change()
        df['rv20_chg'] = df['rv_past_20'].pct_change()

        return df

    def create_features(self, df: pd.DataFrame, target_horizons: list = [5, 20]) -> pd.DataFrame:
        """
        完整特征工程流程

        Args:
            df: 原始指数日线数据
            target_horizons: 目标预测窗口列表，默认 [5, 20] (一周和一月)

        Returns:
            DataFrame: 包含所有特征的数据
        """
        print("开始特征工程...")

        # 1. 按日期排序
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 2. 计算收益率
        print("  - 计算收益率...")
        df = self.calculate_returns(df)

        # 3. 计算未来波动率 (目标变量)
        print(f"  - 计算目标变量: {[f'{h}日' for h in target_horizons]}...")
        df = self.calculate_future_rv(df, horizons=target_horizons)

        # 4. 计算过去波动率
        print("  - 计算过去波动率...")
        df = self.calculate_past_rv(df)

        # 5. 计算 Parkinson 方差
        print("  - 计算 Parkinson 方差...")
        df = self.calculate_parkinson_var(df)

        # 6. 计算成交量特征
        print("  - 计算成交量特征...")
        df = self.calculate_volume_features(df)

        # 7. 计算波动率比率
        print("  - 计算波动率比率...")
        df = self.calculate_rv_ratios(df)

        # 8. 计算收益分布特征
        print("  - 计算收益分布特征...")
        df = self.calculate_rolling_distribution_features(df)

        # 9. 删除中间列
        df = df.drop(columns=['log_hl', 'pk_var'], errors='ignore')

        print(f"特征工程完成: {len(df)} 条记录, {len(df.columns)} 个特征")

        return df

    def get_feature_columns(self, df: pd.DataFrame, target_horizons: list = [5, 20]) -> list:
        """
        获取特征列名

        Args:
            df: 特征工程后的数据
            target_horizons: 目标预测窗口列表

        Returns:
            list: 特征列名列表
        """
        drop_cols = ['ts_code', 'trade_date'] + [f'rv_{h}_fut' for h in target_horizons]
        return [col for col in df.columns if col not in drop_cols]


def create_index_features(input_file: str, output_file: str = None,
                          target_horizons: list = [5, 20]) -> pd.DataFrame:
    """
    从数据文件创建特征

    Args:
        input_file: 输入数据文件路径
        output_file: 输出文件路径
        target_horizons: 目标预测窗口列表，默认 [5, 20]

    Returns:
        DataFrame: 包含特征的数据
    """
    fe = IndexRiskFeatureEngineering()

    print(f"读取数据: {input_file}")
    df = pd.read_csv(input_file)
    print(f"原始数据: {len(df)} 条记录")

    # 特征工程
    df = fe.create_features(df, target_horizons)

    # 保存
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_features.csv"

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已保存到: {output_file}")

    return df


if __name__ == "__main__":
    ml_dir = Path(__file__).parent
    df = create_index_features(
        input_file=str(ml_dir / "dataset" / "index_daily_000001_SH_5years.csv"),
        output_file=str(ml_dir / "dataset" / "index_features.csv"),
        target_horizons=[5, 20]  # 一周和一月
    )
