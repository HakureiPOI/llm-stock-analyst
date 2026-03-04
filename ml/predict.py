"""
波动率预测模块 V2
预测目标：下一交易日的 Yang-Zhang 波动率
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .feature_engineering import VolatilityFeatureEngineering
from .baseline_models import ModelMetrics


class VolatilityPredictor:
    """波动率预测器"""

    def __init__(self, model_dir: str = None):
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = Path(__file__).parent / "models"

        self.model = None
        self.feature_cols = None
        self.feature_importance = None
        self.fe = VolatilityFeatureEngineering()

        self._load_model()

    def _load_model(self):
        """加载训练好的模型"""
        model_path = self.model_dir / "volatility_model_lgb.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.feature_cols = data['feature_cols']
        self.feature_importance = data.get('feature_importance')

        print(f"模型加载成功: {len(self.feature_cols)} 个特征")

    def _get_index_data(self, ts_code: str, days: int = 300) -> pd.DataFrame:
        """获取指数日线数据"""
        from tsdata.index import index_data

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")

        print(f"获取 {ts_code} 数据: {start_date_str} 至 {end_date_str}")

        df = index_data.get_index_daily(
            ts_code=ts_code,
            start_date=start_date_str,
            end_date=end_date_str
        )

        if df is None or df.empty:
            raise ValueError(f"未获取到 {ts_code} 的数据")

        print(f"获取到 {len(df)} 条记录")
        return df

    def predict(self, ts_code: str, days: int = 300,
                include_garch: bool = True) -> dict:
        """
        预测下一交易日波动率

        Args:
            ts_code: 指数代码
            days: 获取历史数据天数
            include_garch: 是否使用GARCH特征

        Returns:
            预测结果字典
        """
        print(f"\n{'=' * 60}")
        print(f"开始预测: {ts_code}")
        print(f"预测目标: 下一交易日 Yang-Zhang 波动率")
        print(f"{'=' * 60}")

        # 1. 获取数据
        df = self._get_index_data(ts_code, days)

        # 2. 特征工程
        print("\n特征工程...")
        df_feat = self.fe.create_features(df, include_garch_features=False)

        # 3. 添加GARCH特征 (如果需要)
        if include_garch:
            try:
                from .garch_features import add_garch_features_to_df
                print("添加GARCH特征...")
                df_feat = add_garch_features_to_df(df_feat, min_train_size=100)
            except Exception as e:
                print(f"GARCH特征添加失败: {e}")

        # 4. 准备预测数据
        # 获取最新的有效数据行
        available_features = [col for col in self.feature_cols if col in df_feat.columns]
        df_valid = df_feat.dropna(subset=available_features)

        if len(df_valid) == 0:
            raise ValueError("没有有效的预测数据")

        # 使用最后一行进行预测
        X = df_valid[available_features].iloc[[-1]]
        pred_log = self.model.predict(X)[0]
        pred_vol = np.exp(pred_log)  # 反对数变换

        # 5. 计算历史统计和分位数
        historical_vol = df_feat['yang_zhang_vol'].dropna().values

        percentile = (np.sum(historical_vol < pred_vol) / len(historical_vol)) * 100 if len(historical_vol) > 0 else 50

        historical_stats = {
            'mean': float(np.mean(historical_vol)) if len(historical_vol) > 0 else 0,
            'std': float(np.std(historical_vol)) if len(historical_vol) > 0 else 0,
            'min': float(np.min(historical_vol)) if len(historical_vol) > 0 else 0,
            'max': float(np.max(historical_vol)) if len(historical_vol) > 0 else 0,
            'q25': float(np.percentile(historical_vol, 25)) if len(historical_vol) > 0 else 0,
            'q50': float(np.percentile(historical_vol, 50)) if len(historical_vol) > 0 else 0,
            'q75': float(np.percentile(historical_vol, 75)) if len(historical_vol) > 0 else 0,
        }

        # 6. 计算评估指标 (如果有真实值)
        metrics = None
        if 'target_vol' in df_feat.columns:
            df_eval = df_feat.dropna(subset=['target_vol'] + available_features)
            if len(df_eval) > 10:
                X_eval = df_eval[available_features]
                y_true = df_eval['target_vol'].values
                y_pred = np.exp(self.model.predict(X_eval))
                metrics = ModelMetrics.calculate_all(y_true, y_pred)

        # 7. 构建结果
        result = {
            'ts_code': ts_code,
            'prediction_type': 'next_day_volatility',
            'prediction_target': 'Yang-Zhang volatility',
            'predicted_volatility': float(pred_vol),
            'percentile': {
                'value': round(percentile, 1),
                'label': self._get_percentile_label(percentile),
                'description': self._get_percentile_description(percentile)
            },
            'historical_stats': historical_stats,
            'metrics': metrics,
            'data_period': f"最近{days}天",
            'prediction_date': datetime.now().isoformat(),
        }

        return result

    def _get_percentile_label(self, percentile: float) -> str:
        """获取分位数标签"""
        if percentile < 25:
            return "极低"
        elif percentile < 50:
            return "偏低"
        elif percentile < 75:
            return "中等"
        elif percentile < 90:
            return "偏高"
        else:
            return "极高"

    def _get_percentile_description(self, percentile: float) -> str:
        """获取分位数描述"""
        if percentile < 25:
            return "市场波动率处于历史低位，风险较小"
        elif percentile < 50:
            return "市场波动率低于历史中位数，风险较低"
        elif percentile < 75:
            return "市场波动率处于历史中等水平"
        elif percentile < 90:
            return "市场波动率高于历史中位数，需关注风险"
        else:
            return "市场波动率处于历史高位，风险较大"

    def print_result(self, result: dict):
        """打印预测结果"""
        print(f"\n预测结果: {result['ts_code']}")
        print(f"{'=' * 60}")

        print(f"\n预测目标: {result['prediction_target']}")
        print(f"\n【下一交易日波动率预测】")
        print(f"  预测值: {result['predicted_volatility']:.6f}")

        # 分位数语义
        p = result['percentile']
        print(f"  历史分位数: {p['value']}% ({p['label']})")
        print(f"  语义解读: {p['description']}")

        # 历史统计
        stats = result['historical_stats']
        print(f"\n历史波动率统计:")
        print(f"  范围: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  均值: {stats['mean']:.6f}")
        print(f"  中位数: {stats['q50']:.6f}")
        print(f"  25%-75%分位: [{stats['q25']:.6f}, {stats['q75']:.6f}]")

        # 模型性能
        if result.get('metrics'):
            metrics = result['metrics']
            print(f"\n模型性能 (历史回测):")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  方向准确率: {metrics['direction_accuracy']:.1f}%")

        return result


def predict_volatility(ts_code: str, days: int = 300,
                       model_dir: str = None,
                       include_garch: bool = True) -> dict:
    """
    便捷函数：预测下一交易日波动率

    Args:
        ts_code: 指数代码
        days: 获取历史数据天数
        model_dir: 模型目录
        include_garch: 是否使用GARCH特征

    Returns:
        预测结果字典
    """
    predictor = VolatilityPredictor(model_dir)
    result = predictor.predict(ts_code, days, include_garch)
    predictor.print_result(result)
    return result


if __name__ == "__main__":
    # 示例
    result = predict_volatility(
        ts_code="000001.SH",
        days=300,
        include_garch=False  # 首次运行可能没有GARCH特征
    )
