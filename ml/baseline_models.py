"""
基线波动率模型
包含 EWMA, GARCH, EGARCH, GJR-GARCH 作为基线对比
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 尝试导入arch库
_ARCH_IMPORT_ERROR = None
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError as e:
    ARCH_AVAILABLE = False
    _ARCH_IMPORT_ERROR = str(e)


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

        # Theil不等系数
        theil_num = np.sqrt(np.mean((y_true - y_pred) ** 2))
        theil_den = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
        theil = theil_num / theil_den if theil_den > 0 else 0

        # 相关系数
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0

        # 命中率
        tolerance = 0.20
        hit_rate = np.mean(np.abs(y_pred - y_true) <= tolerance * y_true_safe) * 100

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy),
            'theil': float(theil),
            'correlation': float(correlation),
            'hit_rate': float(hit_rate)
        }


class EWMABaseline:
    """EWMA (指数加权移动平均) 基线模型"""

    def __init__(self, lambda_: float = 0.94):
        self.lambda_ = lambda_
        self.name = "EWMA"

    def forecast_rolling(self, returns: np.ndarray,
                         min_train: int = 100) -> np.ndarray:
        """
        滚动一步向前预测

        Args:
            returns: 收益率序列
            min_train: 最小训练样本数

        Returns:
            预测波动率序列
        """
        n = len(returns)
        forecasts = np.full(n, np.nan)

        for i in range(min_train, n):
            # 使用历史数据计算EWMA
            hist_ret = returns[:i]
            ewma_var = hist_ret[0] ** 2
            for r in hist_ret[1:]:
                ewma_var = self.lambda_ * ewma_var + (1 - self.lambda_) * r ** 2
            forecasts[i] = np.sqrt(ewma_var)

        return forecasts

    def evaluate(self, returns: np.ndarray, true_vol: np.ndarray,
                 min_train: int = 100) -> dict:
        """评估模型"""
        forecasts = self.forecast_rolling(returns, min_train)
        metrics = ModelMetrics.calculate_all(true_vol, forecasts)
        metrics['model'] = self.name
        return metrics


class GARCHBaseline:
    """GARCH(1,1) 基线模型"""

    def __init__(self):
        self.name = "GARCH(1,1)"
        self._error_msg = None

    def forecast_rolling(self, returns: np.ndarray,
                         min_train: int = 200,
                         refit_freq: int = 20) -> np.ndarray:
        """
        滚动一步向前预测

        Args:
            returns: 收益率序列
            min_train: 最小训练样本数
            refit_freq: 重拟合频率

        Returns:
            预测波动率序列
        """
        if not ARCH_AVAILABLE:
            self._error_msg = f"arch库未安装: {_ARCH_IMPORT_ERROR}"
            return np.full(len(returns), np.nan)

        n = len(returns)
        forecasts = np.full(n, np.nan)
        last_result = None
        fit_errors = 0

        for i in range(min_train, n):
            if i == min_train or (i - min_train) % refit_freq == 0:
                try:
                    model = arch_model(returns[:i] * 100, vol='Garch', p=1, q=1, dist='normal')
                    last_result = model.fit(disp='off', show_warning=False)
                except Exception as e:
                    fit_errors += 1
                    if fit_errors == 1:
                        self._error_msg = f"GARCH拟合错误: {e}"
                    continue

            if last_result is not None:
                try:
                    forecasts[i] = last_result.conditional_volatility[-1] / 100
                except:
                    pass

        return forecasts

    def evaluate(self, returns: np.ndarray, true_vol: np.ndarray,
                 min_train: int = 200) -> dict:
        """评估模型"""
        forecasts = self.forecast_rolling(returns, min_train)
        metrics = ModelMetrics.calculate_all(true_vol, forecasts)
        metrics['model'] = self.name
        if self._error_msg:
            metrics['error'] = self._error_msg
        return metrics


class EGARCHBaseline:
    """EGARCH(1,1) 基线模型 (非对称)"""

    def __init__(self):
        self.name = "EGARCH(1,1)"
        self._error_msg = None

    def forecast_rolling(self, returns: np.ndarray,
                         min_train: int = 200,
                         refit_freq: int = 20) -> np.ndarray:
        """滚动一步向前预测"""
        if not ARCH_AVAILABLE:
            self._error_msg = f"arch库未安装: {_ARCH_IMPORT_ERROR}"
            return np.full(len(returns), np.nan)

        n = len(returns)
        forecasts = np.full(n, np.nan)
        last_result = None
        fit_errors = 0

        for i in range(min_train, n):
            if i == min_train or (i - min_train) % refit_freq == 0:
                try:
                    model = arch_model(returns[:i] * 100, vol='EGARCH', p=1, q=1, dist='normal')
                    last_result = model.fit(disp='off', show_warning=False)
                except Exception as e:
                    fit_errors += 1
                    if fit_errors == 1:
                        self._error_msg = f"EGARCH拟合错误: {e}"
                    continue

            if last_result is not None:
                try:
                    forecasts[i] = last_result.conditional_volatility[-1] / 100
                except:
                    pass

        return forecasts

    def evaluate(self, returns: np.ndarray, true_vol: np.ndarray,
                 min_train: int = 200) -> dict:
        """评估模型"""
        forecasts = self.forecast_rolling(returns, min_train)
        metrics = ModelMetrics.calculate_all(true_vol, forecasts)
        metrics['model'] = self.name
        if self._error_msg:
            metrics['error'] = self._error_msg
        return metrics


class GJRGARCHBaseline:
    """GJR-GARCH(1,1) 基线模型 (非对称)"""

    def __init__(self):
        self.name = "GJR-GARCH(1,1)"
        self._error_msg = None

    def forecast_rolling(self, returns: np.ndarray,
                         min_train: int = 200,
                         refit_freq: int = 20) -> np.ndarray:
        """滚动一步向前预测"""
        if not ARCH_AVAILABLE:
            self._error_msg = f"arch库未安装: {_ARCH_IMPORT_ERROR}"
            return np.full(len(returns), np.nan)

        n = len(returns)
        forecasts = np.full(n, np.nan)
        last_result = None
        fit_errors = 0

        for i in range(min_train, n):
            if i == min_train or (i - min_train) % refit_freq == 0:
                try:
                    model = arch_model(returns[:i] * 100, vol='GARCH', p=1, q=1, o=1, dist='normal')
                    last_result = model.fit(disp='off', show_warning=False)
                except Exception as e:
                    fit_errors += 1
                    if fit_errors == 1:
                        self._error_msg = f"GJR-GARCH拟合错误: {e}"
                    continue

            if last_result is not None:
                try:
                    forecasts[i] = last_result.conditional_volatility[-1] / 100
                except:
                    pass

        return forecasts

    def evaluate(self, returns: np.ndarray, true_vol: np.ndarray,
                 min_train: int = 200) -> dict:
        """评估模型"""
        forecasts = self.forecast_rolling(returns, min_train)
        metrics = ModelMetrics.calculate_all(true_vol, forecasts)
        metrics['model'] = self.name
        if self._error_msg:
            metrics['error'] = self._error_msg
        return metrics


class BaselineComparator:
    """基线模型比较器"""

    def __init__(self):
        self.baselines = [
            EWMABaseline(),
            GARCHBaseline(),
            EGARCHBaseline(),
            GJRGARCHBaseline(),
        ]

    def compare_all(self, returns: np.ndarray, true_vol: np.ndarray,
                    min_train: int = 200) -> pd.DataFrame:
        """
        比较所有基线模型

        Args:
            returns: 收益率序列
            true_vol: 真实波动率序列
            min_train: 最小训练样本数

        Returns:
            DataFrame: 模型比较结果
        """
        results = []

        for model in self.baselines:
            print(f"评估 {model.name}...")
            try:
                metrics = model.evaluate(returns, true_vol, min_train)
                results.append(metrics)
            except Exception as e:
                print(f"  失败: {e}")

        df = pd.DataFrame(results)

        if len(df) > 0:
            # 排序
            df = df.sort_values('rmse')

        return df

    def print_comparison(self, df: pd.DataFrame):
        """打印比较结果"""
        print("\n" + "=" * 80)
        print("基线模型比较结果")
        print("=" * 80)

        if len(df) == 0:
            print("无有效结果")
            return

        # 格式化输出
        cols = ['model', 'mae', 'rmse', 'r2', 'mape', 'direction_accuracy', 'theil', 'correlation', 'error']
        df_display = df[[c for c in cols if c in df.columns]]

        for _, row in df_display.iterrows():
            print(f"\n{row['model']}")

            # 如果有错误信息，优先显示
            if 'error' in row and pd.notna(row.get('error')):
                print(f"  错误: {row['error']}")
            elif pd.isna(row.get('mae')):
                print(f"  (无有效预测结果)")
            else:
                print(f"  MAE: {row['mae']:.6f}")
                print(f"  RMSE: {row['rmse']:.6f}")
                print(f"  R²: {row['r2']:.4f}")
                print(f"  MAPE: {row['mape']:.2f}%")
                print(f"  方向准确率: {row['direction_accuracy']:.1f}%")
                print(f"  Theil系数: {row['theil']:.4f}")
                print(f"  相关系数: {row['correlation']:.4f}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    # 测试
    from pathlib import Path

    ml_dir = Path(__file__).parent

    # 读取数据
    df = pd.read_csv(ml_dir / "dataset" / "index_features.csv")

    # 准备数据
    returns = df['log_ret'].values
    true_vol = df['target_vol'].values  # Yang-Zhang波动率

    # 比较
    comparator = BaselineComparator()
    results = comparator.compare_all(returns, true_vol, min_train=200)
    comparator.print_comparison(results)

    # 保存结果
    results.to_csv(ml_dir / "models" / "baseline_comparison.csv", index=False)
