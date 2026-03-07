"""
基线波动率模型
所有模型都预测 Yang-Zhang 波动率，确保公平比较
包含：Naive, MA, EWMA
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelMetrics:
    """模型评价指标"""

    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """计算所有评价指标"""
        # 去除NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {}

        # 基础指标
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # MAPE
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

        # 方向准确率
        if len(y_true) > 1:
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            direction_accuracy = np.mean(direction_true == direction_pred) * 100
        else:
            direction_accuracy = 0

        # 相关系数
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0

        # 命中率 (±20%)
        tolerance = 0.20
        hit_rate = np.mean(np.abs(y_pred - y_true) <= tolerance * y_true_safe) * 100

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy),
            'correlation': float(correlation),
            'hit_rate': float(hit_rate)
        }


class NaiveBaseline:
    """
    Naive (Persistence) 基线模型
    预测 = 前一日的 Yang-Zhang 波动率
    """

    def __init__(self):
        self.name = "Naive"

    def forecast_rolling(self, yz_vol: np.ndarray,
                         min_train: int = 20) -> np.ndarray:
        """滚动一步向前预测"""
        n = len(yz_vol)
        forecasts = np.full(n, np.nan)

        # 找到第一个有效值的位置
        first_valid = np.where(~np.isnan(yz_vol))[0]
        if len(first_valid) == 0:
            return forecasts
        start_idx = max(first_valid[0] + 1, min_train)

        for i in range(start_idx, n):
            # 找到最近的有效值
            for j in range(i - 1, -1, -1):
                if not np.isnan(yz_vol[j]):
                    forecasts[i] = yz_vol[j]
                    break

        return forecasts

    def evaluate(self, yz_vol: np.ndarray, true_vol: np.ndarray,
                 min_train: int = 20) -> dict:
        """评估模型"""
        forecasts = self.forecast_rolling(yz_vol, min_train)
        metrics = ModelMetrics.calculate_all(true_vol, forecasts)
        metrics['model'] = self.name
        return metrics


class MABaseline:
    """
    Moving Average 基线模型
    预测 = 过去 N 日的 Yang-Zhang 波动率均值
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.name = f"MA-{window}"

    def forecast_rolling(self, yz_vol: np.ndarray,
                         min_train: int = 20) -> np.ndarray:
        """滚动一步向前预测"""
        n = len(yz_vol)
        forecasts = np.full(n, np.nan)

        first_valid = np.where(~np.isnan(yz_vol))[0]
        if len(first_valid) == 0:
            return forecasts
        start_idx = max(first_valid[0] + self.window, min_train)

        for i in range(start_idx, n):
            hist_vol = yz_vol[i - self.window:i]
            valid_vol = hist_vol[~np.isnan(hist_vol)]
            if len(valid_vol) > 0:
                forecasts[i] = np.mean(valid_vol)

        return forecasts

    def evaluate(self, yz_vol: np.ndarray, true_vol: np.ndarray,
                 min_train: int = 20) -> dict:
        """评估模型"""
        forecasts = self.forecast_rolling(yz_vol, min_train)
        metrics = ModelMetrics.calculate_all(true_vol, forecasts)
        metrics['model'] = self.name
        return metrics


class EWMABaseline:
    """
    EWMA (指数加权移动平均) 基线模型
    直接对 Yang-Zhang 波动率序列做 EWMA
    """

    def __init__(self, lambda_: float = 0.94):
        self.lambda_ = lambda_
        self.name = f"EWMA-λ{lambda_}"

    def forecast_rolling(self, yz_vol: np.ndarray,
                         min_train: int = 20) -> np.ndarray:
        """滚动一步向前预测"""
        n = len(yz_vol)
        forecasts = np.full(n, np.nan)

        first_valid = np.where(~np.isnan(yz_vol))[0]
        if len(first_valid) == 0:
            return forecasts
        start_idx = max(first_valid[0] + 1, min_train)

        for i in range(start_idx, n):
            hist_vol = yz_vol[:i]
            valid_vol = hist_vol[~np.isnan(hist_vol)]

            if len(valid_vol) == 0:
                continue

            ewma_val = valid_vol[0]
            for v in valid_vol[1:]:
                ewma_val = self.lambda_ * ewma_val + (1 - self.lambda_) * v
            forecasts[i] = ewma_val

        return forecasts

    def evaluate(self, yz_vol: np.ndarray, true_vol: np.ndarray,
                 min_train: int = 20) -> dict:
        """评估模型"""
        forecasts = self.forecast_rolling(yz_vol, min_train)
        metrics = ModelMetrics.calculate_all(true_vol, forecasts)
        metrics['model'] = self.name
        return metrics


class BaselineComparator:
    """基线模型比较器"""

    def __init__(self):
        self.baselines = [
            NaiveBaseline(),
            MABaseline(window=5),
            MABaseline(window=20),
            EWMABaseline(lambda_=0.94),
        ]

    def compare_all(self, yz_vol: np.ndarray, true_vol: np.ndarray,
                    min_train: int = 20) -> pd.DataFrame:
        """比较所有基线模型"""
        results = []

        for model in self.baselines:
            print(f"  评估 {model.name}...")
            try:
                metrics = model.evaluate(yz_vol, true_vol, min_train)
                results.append(metrics)
            except Exception as e:
                print(f"    失败: {e}")

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('mae')
        return df

    def print_comparison(self, df: pd.DataFrame):
        """打印比较结果"""
        print("\n" + "=" * 70)
        print("基线模型比较结果（预测 Yang-Zhang 波动率）")
        print("=" * 70)

        cols = ['model', 'mae', 'rmse', 'r2', 'mape', 'direction_accuracy', 'correlation', 'hit_rate']
        if len(df) > 0:
            print(df[[c for c in cols if c in df.columns]].to_string(index=False))
        else:
            print("无有效结果")


if __name__ == "__main__":
    from pathlib import Path

    ml_dir = Path(__file__).parent
    df = pd.read_csv(ml_dir / "dataset" / "index_features.csv")

    yz_vol = df['yang_zhang_vol'].values
    true_vol = df['target_vol'].values

    comparator = BaselineComparator()
    results = comparator.compare_all(yz_vol, true_vol, min_train=20)
    comparator.print_comparison(results)

    results.to_csv(ml_dir / "models" / "model_comparison_yz.csv", index=False)
    print(f"\n结果已保存")
