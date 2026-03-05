"""
波动率预测模块
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


# SHAP解释器（可选）
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


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
        self.shap_explainer = None
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

        # 初始化SHAP解释器
        if SHAP_AVAILABLE:
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
            except Exception:
                pass

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

    def _analyze_shap(self, X: pd.DataFrame, top_n: int = 5) -> Dict:
        """SHAP特征贡献度分析，返回原始数据"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return None

        try:
            shap_values = self.shap_explainer.shap_values(X)

            # 获取特征贡献度
            contributions = []
            for i, col in enumerate(X.columns):
                contributions.append({
                    'feature': col,
                    'value': float(X.iloc[0, i]),
                    'shap_value': float(shap_values[0, i]),
                    'contribution_pct': 0  # 后面计算
                })

            # 按绝对贡献排序
            contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

            # 计算贡献百分比
            total_shap = sum(abs(c['shap_value']) for c in contributions)
            for c in contributions:
                c['contribution_pct'] = abs(c['shap_value']) / total_shap * 100 if total_shap > 0 else 0

            return {
                'top_features': contributions[:top_n]
            }

        except Exception as e:
            print(f"SHAP分析失败: {e}")
            return None

    def predict(self, ts_code: str, days: int = 300,
                include_garch: bool = True,
                include_shap: bool = True) -> dict:
        """
        预测下一交易日波动率，返回原始数据
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

        # 3. 添加GARCH特征
        if include_garch:
            try:
                from .garch_features import add_garch_features_to_df
                print("添加GARCH特征...")
                df_feat = add_garch_features_to_df(df_feat, min_train_size=100)
            except Exception as e:
                print(f"GARCH特征添加失败: {e}")

        # 4. 准备预测数据
        available_features = [col for col in self.feature_cols if col in df_feat.columns]
        df_valid = df_feat.dropna(subset=available_features)

        if len(df_valid) == 0:
            raise ValueError("没有有效的预测数据")

        # 使用最后一行进行预测
        X = df_valid[available_features].iloc[[-1]]
        pred_log = self.model.predict(X)[0]
        pred_vol = np.exp(pred_log)

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
            'count': len(historical_vol)
        }

        # 6. 波动率变化趋势
        volatility_trend = None
        if len(df_valid) >= 2:
            prev_vol = df_valid['yang_zhang_vol'].iloc[-2]
            curr_vol = df_valid['yang_zhang_vol'].iloc[-1]
            change_pct = (curr_vol - prev_vol) / prev_vol * 100 if prev_vol > 0 else 0
            volatility_trend = {
                'previous_vol': float(prev_vol),
                'current_vol': float(curr_vol),
                'change_pct': round(change_pct, 2)
            }

        # 7. SHAP分析
        shap_result = None
        if include_shap:
            shap_result = self._analyze_shap(X, top_n=5)

        # 8. 计算评估指标
        metrics = None
        if 'target_vol' in df_feat.columns:
            df_eval = df_feat.dropna(subset=['target_vol'] + available_features)
            if len(df_eval) > 10:
                X_eval = df_eval[available_features]
                y_true = df_eval['target_vol'].values
                y_pred = np.exp(self.model.predict(X_eval))
                metrics = ModelMetrics.calculate_all(y_true, y_pred)

        # 9. 构建结果 - 只返回原始数据
        result = {
            'ts_code': ts_code,
            'prediction_type': 'next_day_volatility',
            'prediction_target': 'Yang-Zhang volatility',
            'predicted_volatility': float(pred_vol),
            'volatility_pct': f"{pred_vol * 100:.2f}%",
            'percentile': round(percentile, 1),  # 只返回数值
            'historical_stats': historical_stats,
            'volatility_trend': volatility_trend,
            'shap_analysis': shap_result,
            'metrics': metrics,
            'data_period': f"最近{days}天",
            'prediction_date': datetime.now().isoformat(),
        }

        return result

    def print_result(self, result: dict):
        """打印预测结果"""
        print(f"\n预测结果: {result['ts_code']}")
        print(f"{'=' * 60}")
        print(f"预测波动率: {result['predicted_volatility']:.6f} ({result['volatility_pct']})")
        print(f"历史分位数: {result['percentile']}%")

        if result.get('volatility_trend'):
            trend = result['volatility_trend']
            print(f"波动率趋势: {trend['change_pct']:.2f}% vs 前一日")

        if result.get('shap_analysis'):
            print(f"\nTop 5 影响特征:")
            for feat in result['shap_analysis']['top_features']:
                direction = "+" if feat['shap_value'] > 0 else "-"
                print(f"  {feat['feature']}: {direction} {feat['contribution_pct']:.1f}%")

        return result


def predict_volatility(ts_code: str, days: int = 300,
                       model_dir: str = None,
                       include_garch: bool = True,
                       include_shap: bool = True) -> dict:
    """便捷函数：预测下一交易日波动率"""
    predictor = VolatilityPredictor(model_dir)
    result = predictor.predict(ts_code, days, include_garch, include_shap)
    predictor.print_result(result)
    return result


if __name__ == "__main__":
    result = predict_volatility(
        ts_code="000001.SH",
        days=300,
        include_garch=True,
        include_shap=True
    )
