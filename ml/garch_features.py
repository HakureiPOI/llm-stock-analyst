"""
GARCH特征提取模块
从多个GARCH家族模型提取特征
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("警告: arch库未安装，GARCH特征将不可用。请运行: pip install arch")


class GARCHFeatureExtractor:
    """GARCH特征提取器"""

    def __init__(self):
        self.models = {}
        self.fitted_results = {}

    def fit_garch(self, returns: np.ndarray, name: str = 'garch',
                  p: int = 1, q: int = 1) -> Dict:
        """
        拟合标准 GARCH(p,q) 模型

        Args:
            returns: 收益率序列
            name: 模型名称
            p: GARCH阶数
            q: ARCH阶数

        Returns:
            拟合结果字典
        """
        if not ARCH_AVAILABLE:
            return {}

        try:
            model = arch_model(returns * 100, vol='Garch', p=p, q=q, dist='normal')
            result = model.fit(disp='off', show_warning=False)

            self.models[name] = model
            self.fitted_results[name] = result

            return {
                'name': name,
                'type': 'GARCH',
                'conditional_vol': result.conditional_volatility / 100,  # 还原缩放
                'residuals': result.resid / 100,
                'std_residuals': result.resid / result.conditional_volatility,
                'params': result.params.to_dict(),
                'log_likelihood': result.loglikelihood,
                'aic': result.aic,
                'bic': result.bic,
            }
        except Exception as e:
            print(f"GARCH拟合失败: {e}")
            return {}

    def fit_egarch(self, returns: np.ndarray, name: str = 'egarch',
                   p: int = 1, q: int = 1) -> Dict:
        """
        拟合 EGARCH(p,q) 模型 (非对称)

        Args:
            returns: 收益率序列
            name: 模型名称
            p: GARCH阶数
            q: ARCH阶数

        Returns:
            拟合结果字典
        """
        if not ARCH_AVAILABLE:
            return {}

        try:
            model = arch_model(returns * 100, vol='EGARCH', p=p, q=q, dist='normal')
            result = model.fit(disp='off', show_warning=False)

            self.models[name] = model
            self.fitted_results[name] = result

            return {
                'name': name,
                'type': 'EGARCH',
                'conditional_vol': result.conditional_volatility / 100,
                'residuals': result.resid / 100,
                'std_residuals': result.resid / result.conditional_volatility,
                'params': result.params.to_dict(),
                'log_likelihood': result.loglikelihood,
                'aic': result.aic,
                'bic': result.bic,
            }
        except Exception as e:
            print(f"EGARCH拟合失败: {e}")
            return {}

    def fit_gjr_garch(self, returns: np.ndarray, name: str = 'gjr_garch',
                      p: int = 1, q: int = 1) -> Dict:
        """
        拟合 GJR-GARCH(p,q) 模型 (非对称)

        Args:
            returns: 收益率序列
            name: 模型名称
            p: GARCH阶数
            q: ARCH阶数

        Returns:
            拟合结果字典
        """
        if not ARCH_AVAILABLE:
            return {}

        try:
            model = arch_model(returns * 100, vol='GARCH', p=p, q=q, o=1, dist='normal')
            result = model.fit(disp='off', show_warning=False)

            self.models[name] = model
            self.fitted_results[name] = result

            return {
                'name': name,
                'type': 'GJR-GARCH',
                'conditional_vol': result.conditional_volatility / 100,
                'residuals': result.resid / 100,
                'std_residuals': result.resid / result.conditional_volatility,
                'params': result.params.to_dict(),
                'log_likelihood': result.loglikelihood,
                'aic': result.aic,
                'bic': result.bic,
            }
        except Exception as e:
            print(f"GJR-GARCH拟合失败: {e}")
            return {}

    def fit_all_models(self, returns: np.ndarray) -> Dict:
        """
        拟合所有GARCH模型

        Args:
            returns: 收益率序列

        Returns:
            所有模型结果字典
        """
        results = {}

        # GARCH(1,1)
        r = self.fit_garch(returns, 'garch_11')
        if r:
            results['garch_11'] = r

        # EGARCH(1,1)
        r = self.fit_egarch(returns, 'egarch_11')
        if r:
            results['egarch_11'] = r

        # GJR-GARCH(1,1)
        r = self.fit_gjr_garch(returns, 'gjr_garch_11')
        if r:
            results['gjr_garch_11'] = r

        return results

    def extract_features(self, returns: np.ndarray,
                         min_train_size: int = 500) -> pd.DataFrame:
        """
        提取GARCH特征 (用于时间序列预测)

        使用滚动窗口方式，每个时刻用历史数据拟合模型并提取特征

        Args:
            returns: 收益率序列
            min_train_size: 最小训练样本数

        Returns:
            DataFrame: GARCH特征
        """
        if not ARCH_AVAILABLE:
            return pd.DataFrame()

        n = len(returns)
        features = {
            'garch_vol': np.full(n, np.nan),
            'garch_resid': np.full(n, np.nan),
            'garch_std_resid': np.full(n, np.nan),
            'egarch_vol': np.full(n, np.nan),
            'egarch_resid': np.full(n, np.nan),
            'egarch_std_resid': np.full(n, np.nan),
            'gjr_vol': np.full(n, np.nan),
            'gjr_resid': np.full(n, np.nan),
            'gjr_std_resid': np.full(n, np.nan),
        }

        print("提取GARCH特征 (滚动窗口)...")

        # 滚动拟合 (为了效率，每隔一定步长重新拟合)
        refit_freq = 20  # 每20天重新拟合一次

        for i in range(min_train_size, n):
            # 判断是否需要重新拟合
            if i == min_train_size or (i - min_train_size) % refit_freq == 0:
                train_data = returns[:i]

                try:
                    # 拟合三个模型
                    garch = arch_model(train_data * 100, vol='Garch', p=1, q=1, dist='normal')
                    garch_result = garch.fit(disp='off', show_warning=False)

                    egarch = arch_model(train_data * 100, vol='EGARCH', p=1, q=1, dist='normal')
                    egarch_result = egarch.fit(disp='off', show_warning=False)

                    gjr = arch_model(train_data * 100, vol='GARCH', p=1, q=1, o=1, dist='normal')
                    gjr_result = gjr.fit(disp='off', show_warning=False)

                except Exception as e:
                    continue

            # 使用最近的拟合结果计算特征
            try:
                # GARCH特征
                features['garch_vol'][i] = garch_result.conditional_volatility[-1] / 100
                features['garch_resid'][i] = garch_result.resid[-1] / 100
                features['garch_std_resid'][i] = garch_result.resid[-1] / garch_result.conditional_volatility[-1]

                # EGARCH特征
                features['egarch_vol'][i] = egarch_result.conditional_volatility[-1] / 100
                features['egarch_resid'][i] = egarch_result.resid[-1] / 100
                features['egarch_std_resid'][i] = egarch_result.resid[-1] / egarch_result.conditional_volatility[-1]

                # GJR-GARCH特征
                features['gjr_vol'][i] = gjr_result.conditional_volatility[-1] / 100
                features['gjr_resid'][i] = gjr_result.resid[-1] / 100
                features['gjr_std_resid'][i] = gjr_result.resid[-1] / gjr_result.conditional_volatility[-1]
            except:
                pass

            if i % 100 == 0:
                print(f"  处理进度: {i}/{n}")

        return pd.DataFrame(features)


class EWMAVolatility:
    """EWMA (指数加权移动平均) 波动率"""

    def __init__(self, lambda_: float = 0.94):
        """
        Args:
            lambda_: 衰减因子，RiskMetrics推荐0.94 (日频)
        """
        self.lambda_ = lambda_

    def fit(self, returns: np.ndarray) -> np.ndarray:
        """
        计算EWMA波动率

        Args:
            returns: 收益率序列

        Returns:
            EWMA波动率序列
        """
        n = len(returns)
        ewma_var = np.zeros(n)
        ewma_var[0] = returns[0] ** 2

        for i in range(1, n):
            ewma_var[i] = self.lambda_ * ewma_var[i-1] + (1 - self.lambda_) * returns[i] ** 2

        return np.sqrt(ewma_var)

    def forecast(self, returns: np.ndarray) -> float:
        """
        一步向前预测

        Args:
            returns: 收益率序列

        Returns:
            预测的下期波动率
        """
        ewma_vol = self.fit(returns)
        return ewma_vol[-1]


def add_garch_features_to_df(df: pd.DataFrame,
                             min_train_size: int = 500) -> pd.DataFrame:
    """
    为DataFrame添加GARCH特征

    Args:
        df: 包含 'log_ret' 列的DataFrame
        min_train_size: 最小训练样本数

    Returns:
        添加了GARCH特征的DataFrame
    """
    if not ARCH_AVAILABLE:
        print("arch库未安装，跳过GARCH特征")
        df['garch_vol'] = np.nan
        df['egarch_vol'] = np.nan
        df['gjr_vol'] = np.nan
        return df

    extractor = GARCHFeatureExtractor()
    returns = df['log_ret'].values

    garch_features = extractor.extract_features(returns, min_train_size)

    # 合并到原DataFrame
    for col in garch_features.columns:
        df[col] = garch_features[col].values

    return df


if __name__ == "__main__":
    # 测试
    import pandas as pd
    from pathlib import Path

    ml_dir = Path(__file__).parent

    # 读取特征数据
    df = pd.read_csv(ml_dir / "dataset" / "index_features.csv")

    # 添加GARCH特征
    print("添加GARCH特征...")
    df = add_garch_features_to_df(df, min_train_size=500)

    # 保存
    output_file = ml_dir / "dataset" / "index_features_with_garch.csv"
    df.to_csv(output_file, index=False)
    print(f"已保存到: {output_file}")
