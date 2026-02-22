import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from .feature_engineering import FeatureEngineering


class VolatilityPredictor:
    """波动率预测器"""

    def __init__(self, model_path: str = "ml/models/volatility_model.pkl"):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.fe = FeatureEngineering()
        self._load_model()

    def _load_model(self):
        """加载训练好的模型"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"模型加载成功: {self.model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

    def _get_stock_data(self, ts_code: str, days: int = 500):
        """
        获取股票日线数据

        Args:
            ts_code: 股票代码
            days: 获取天数

        Returns:
            DataFrame: 原始K线数据
        """
        from tsdata.stock import stock_data

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")

        print(f"获取 {ts_code} 数据: {start_date_str} 至 {end_date_str}")

        df = stock_data.get_daily(
            ts_code=ts_code,
            start_date=start_date_str,
            end_date=end_date_str
        )

        if df is None or df.empty:
            raise ValueError(f"未获取到 {ts_code} 的数据")

        print(f"获取到 {len(df)} 条记录")
        return df

    def _prepare_features(self, df: pd.DataFrame, window_size: int = 5):
        """
        准备特征(不分割数据,保留所有行用于预测)

        Args:
            df: 原始数据
            window_size: 波动率窗口

        Returns:
            DataFrame: 包含特征的数据(可能含NaN)
        """
        # 计算未来波动率(目标,仅用于特征工程过程)
        df = self.fe.calculate_future_volatility(df, window_size)

        # 添加特征
        df = self.fe.add_features(df)

        return df

    def predict(self, ts_code: str, days: int = 500, window_size: int = 5):
        """
        预测股票未来波动率

        Args:
            ts_code: 股票代码
            days: 获取天数
            window_size: 波动率计算窗口

        Returns:
            dict: 预测结果
        """
        print(f"\n{'='*60}")
        print(f"开始预测: {ts_code}")
        print(f"{'='*60}")

        # 1. 获取数据
        df = self._get_stock_data(ts_code, days)

        # 2. 特征工程
        print("\n特征工程...")
        df_feat = self._prepare_features(df, window_size)

        # 3. 准备预测数据(移除不需要的列)
        columns_to_drop = ['ts_code', 'trade_date', 'log_return', 'future_volatility']
        feature_cols = [c for c in df_feat.columns if c not in columns_to_drop]

        X = df_feat[feature_cols].dropna()

        if len(X) == 0:
            raise ValueError("没有有效的预测数据(全部含NaN)")

        print(f"有效预测样本: {len(X)}")

        # 4. 预测
        print("\n预测...")
        predictions = self.model.predict(X)

        # 5. 计算模型表现(对比预测与实际波动率)
        # 找出有实际波动率且特征完整的行
        df_valid = df_feat.dropna(subset=['future_volatility'])
        common_index = df_valid.index.intersection(X.index)

        # 获取对应的预测值
        predictions_aligned = predictions[X.index.get_indexer(common_index)]

        # 计算指标
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        actual = df_valid.loc[common_index, 'future_volatility'].values

        if len(actual) > 0:
            mae = mean_absolute_error(actual, predictions_aligned)
            mse = mean_squared_error(actual, predictions_aligned)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual, predictions_aligned)
        else:
            mae = mse = rmse = r2 = np.nan

        # 6. 构建结果
        df_pred = df_feat.loc[common_index].copy()
        df_pred['predicted_volatility'] = predictions_aligned

        result = {
            'ts_code': ts_code,
            'data_period': f"最近{days}天",
            'total_records': len(df),
            'valid_predictions': len(actual),
            'predictions': df_pred[['trade_date', 'predicted_volatility']].to_dict('records'),
            'metrics': {
                'mae': float(mae) if not np.isnan(mae) else None,
                'mse': float(mse) if not np.isnan(mse) else None,
                'rmse': float(rmse) if not np.isnan(rmse) else None,
                'r2': float(r2) if not np.isnan(r2) else None
            },
            'actual_vs_predicted': list(zip(
                df_valid.loc[common_index, 'trade_date'].dt.strftime('%Y-%m-%d'),
                actual.tolist(),
                predictions_aligned.tolist()
            ))
        }

        return result

    def print_result(self, result: dict):
        """打印预测结果"""
        print(f"\n预测结果: {result['ts_code']}")
        print(f"{'='*60}")

        print(f"\n数据概况:")
        print(f"  时间范围: {result['data_period']}")
        print(f"  总记录数: {result['total_records']}")
        print(f"  有效预测数: {result['valid_predictions']}")

        print(f"\n模型性能:")
        metrics = result['metrics']
        print(f"  MAE:  {metrics['mae']:.6f}" if metrics['mae'] is not None else "  MAE:  N/A")
        print(f"  MSE:  {metrics['mse']:.6f}" if metrics['mse'] is not None else "  MSE:  N/A")
        print(f"  RMSE: {metrics['rmse']:.6f}" if metrics['rmse'] is not None else "  RMSE: N/A")
        print(f"  R²:   {metrics['r2']:.6f}" if metrics['r2'] is not None else "  R²:   N/A")

        if result['actual_vs_predicted']:
            print(f"\n最近10天预测对比:")
            print(f"{'日期':<12} {'实际':<12} {'预测':<12} {'误差':<12}")
            for date, actual, pred in result['actual_vs_predicted'][-10:]:
                error = abs(actual - pred) if not np.isnan(actual) and not np.isnan(pred) else np.nan
                print(f"{date:<12} {actual:<12.6f} {pred:<12.6f} {error:<12.6f}")

        return result


def predict_stock(ts_code: str, days: int = 500, window_size: int = 5,
                model_path: str = None):
    """
    便捷函数: 预测单只股票

    Args:
        ts_code: 股票代码
        days: 获取天数
        window_size: 波动率窗口
        model_path: 模型路径

    Returns:
        dict: 预测结果
    """
    if model_path is None:
        model_path = "ml/models/volatility_model.pkl"

    predictor = VolatilityPredictor(model_path)
    result = predictor.predict(ts_code, days, window_size)
    predictor.print_result(result)

    return result


def predict_batch(stock_codes: list, days: int = 500, window_size: int = 5,
                model_path: str = None):
    """
    便捷函数: 批量预测多只股票

    Args:
        stock_codes: 股票代码列表
        days: 获取天数
        window_size: 波动率窗口
        model_path: 模型路径

    Returns:
        list: 预测结果列表
    """
    if model_path is None:
        model_path = "ml/models/volatility_model.pkl"

    predictor = VolatilityPredictor(model_path)
    results = []

    for i, code in enumerate(stock_codes, 1):
        print(f"\n\n进度: {i}/{len(stock_codes)}")
        try:
            result = predictor.predict(code, days, window_size)
            results.append(result)
        except Exception as e:
            print(f"预测 {code} 失败: {e}")
            results.append({'ts_code': code, 'error': str(e)})

    # 汇总
    print(f"\n\n{'='*60}")
    print("批量预测完成!")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if 'metrics' in r and r['metrics']['mae'] is not None)
    print(f"\n成功: {success_count}/{len(stock_codes)}")

    return results


if __name__ == "__main__":
    # 示例: 预测单只股票
    print("示例1: 预测单只股票")
    result = predict_stock(
        ts_code="600519.SH",  # 贵州茅台
        days=365,
        window_size=5
    )

    # 示例2: 批量预测
    # print("\n\n示例2: 批量预测")
    # codes = ["600519.SH", "601318.SH", "600036.SH"]
    # predict_batch(codes, days=365, window_size=5)
