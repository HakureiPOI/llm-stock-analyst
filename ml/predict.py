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

    def _calculate_semantic_metrics(self, df_feat: pd.DataFrame, pred_vol: float) -> Dict:
        """
        计算多种语义指标，更全面地评估波动率状态
        
        包括：
        1. 滚动分位数（不同窗口）
        2. 相对近期均值的偏离度
        3. 波动率趋势指标
        4. 波动率稳定性指标
        5. 波动率聚类状态
        """
        vol_series = df_feat['yang_zhang_vol'].dropna()
        n = len(vol_series)
        
        if n < 20:
            return {}
        
        metrics = {}
        
        # ==================== 1. 分位数指标 ====================
        # 全局分位数
        global_pct = (np.sum(vol_series < pred_vol) / n) * 100
        
        # 滚动分位数（近1年、近半年、近3个月）
        windows = {
            '1y': min(252, n),
            '6m': min(126, n),
            '3m': min(63, n),
            '1m': min(21, n)
        }
        
        rolling_percentiles = {}
        for name, window in windows.items():
            if n >= window:
                recent_vol = vol_series.iloc[-window:]
                rolling_pct = (np.sum(recent_vol < pred_vol) / window) * 100
                rolling_percentiles[name] = round(rolling_pct, 1)
        
        metrics['percentile_global'] = round(global_pct, 1)
        metrics['percentile_rolling'] = rolling_percentiles
        
        # ==================== 2. 相对均值偏离度 ====================
        recent_means = {}
        deviations = {}
        
        for name, window in windows.items():
            if n >= window:
                mean_val = vol_series.iloc[-window:].mean()
                recent_means[name] = float(mean_val)
                # 偏离度 = (当前值 - 均值) / 均值 * 100
                deviation = (pred_vol - mean_val) / mean_val * 100 if mean_val > 0 else 0
                deviations[name] = round(deviation, 1)
        
        metrics['recent_means'] = recent_means
        metrics['deviation_from_mean'] = deviations
        
        # ==================== 3. 波动率趋势指标 ====================
        trend_metrics = {}
        
        # 连续上升/下降天数
        vol_changes = vol_series.diff()
        consecutive_up = 0
        consecutive_down = 0
        
        for change in vol_changes.iloc[::-1]:
            if change > 0:
                consecutive_up += 1
                if consecutive_down > 0:
                    break
            elif change < 0:
                consecutive_down += 1
                if consecutive_up > 0:
                    break
            else:
                break
        
        trend_metrics['consecutive_up_days'] = consecutive_up
        trend_metrics['consecutive_down_days'] = consecutive_down
        
        # 近5日/10日/20日变化趋势
        for period in [5, 10, 20]:
            if n >= period:
                period_ago = vol_series.iloc[-period]
                change = (vol_series.iloc[-1] - period_ago) / period_ago * 100 if period_ago > 0 else 0
                trend_metrics[f'{period}d_change_pct'] = round(change, 1)
        
        metrics['trend'] = trend_metrics
        
        # ==================== 4. 波动率稳定性指标 ====================
        stability_metrics = {}
        
        # 近期波动率的变异系数 (CV = std / mean)
        for name, window in windows.items():
            if n >= window:
                recent = vol_series.iloc[-window:]
                mean_val = recent.mean()
                std_val = recent.std()
                cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                stability_metrics[f'cv_{name}'] = round(cv, 1)
        
        # 波动率范围（近1年）
        if n >= windows['1y']:
            recent_1y = vol_series.iloc[-windows['1y']:]
            stability_metrics['range_1y_min'] = float(recent_1y.min())
            stability_metrics['range_1y_max'] = float(recent_1y.max())
            stability_metrics['range_1y_ratio'] = round(recent_1y.max() / recent_1y.min(), 2) if recent_1y.min() > 0 else 0
        
        metrics['stability'] = stability_metrics
        
        # ==================== 5. 波动率聚类状态 ====================
        # 判断当前处于高波动还是低波动阶段
        cluster_metrics = {}
        
        if n >= 60:
            # 使用近60日均值作为分界线
            recent_60 = vol_series.iloc[-60:]
            mean_60 = recent_60.mean()
            
            # 统计近20日高于/低于均值的天数
            recent_20 = vol_series.iloc[-20:]
            above_mean_days = (recent_20 > mean_60).sum()
            below_mean_days = (recent_20 < mean_60).sum()
            
            cluster_metrics['above_mean_days_20d'] = int(above_mean_days)
            cluster_metrics['below_mean_days_20d'] = int(below_mean_days)
            
            # 判断波动状态
            if above_mean_days >= 15:  # 75%以上时间高于均值
                cluster_metrics['regime'] = 'high_volatility'
                cluster_metrics['regime_desc'] = '高波动阶段'
            elif below_mean_days >= 15:  # 75%以上时间低于均值
                cluster_metrics['regime'] = 'low_volatility'
                cluster_metrics['regime_desc'] = '低波动阶段'
            else:
                cluster_metrics['regime'] = 'transition'
                cluster_metrics['regime_desc'] = '波动转换期'
        
        metrics['cluster'] = cluster_metrics
        
        # ==================== 6. 综合风险等级评估 ====================
        # 基于多个指标综合判断
        risk_signals = []
        
        # 信号1：滚动分位数
        if rolling_percentiles.get('6m', 50) >= 75:
            risk_signals.append(('rolling_pct_high', 1))
        elif rolling_percentiles.get('6m', 50) <= 25:
            risk_signals.append(('rolling_pct_low', -1))
        
        # 信号2：偏离近期均值
        if deviations.get('3m', 0) > 50:  # 高于近3月均值50%以上
            risk_signals.append(('deviation_high', 1))
        elif deviations.get('3m', 0) < -30:  # 低于近3月均值30%以上
            risk_signals.append(('deviation_low', -1))
        
        # 信号3：波动率聚类状态
        if cluster_metrics.get('regime') == 'high_volatility':
            risk_signals.append(('regime_high', 1))
        elif cluster_metrics.get('regime') == 'low_volatility':
            risk_signals.append(('regime_low', -1))
        
        # 信号4：连续上升/下降
        if consecutive_up >= 3:
            risk_signals.append(('consecutive_up', 1))
        elif consecutive_down >= 3:
            risk_signals.append(('consecutive_down', -1))
        
        # 计算综合得分
        risk_score = sum(s[1] for s in risk_signals)
        
        if risk_score >= 2:
            metrics['risk_level'] = 'high'
            metrics['risk_level_desc'] = '高风险'
        elif risk_score >= 1:
            metrics['risk_level'] = 'medium_high'
            metrics['risk_level_desc'] = '中高风险'
        elif risk_score <= -2:
            metrics['risk_level'] = 'low'
            metrics['risk_level_desc'] = '低风险'
        elif risk_score <= -1:
            metrics['risk_level'] = 'medium_low'
            metrics['risk_level_desc'] = '中低风险'
        else:
            metrics['risk_level'] = 'medium'
            metrics['risk_level_desc'] = '中等风险'
        
        metrics['risk_signals'] = [s[0] for s in risk_signals]
        metrics['risk_score'] = risk_score
        
        return metrics

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

        # 5. 计算历史统计
        historical_vol = df_feat['yang_zhang_vol'].dropna().values
        
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

        # 9. 计算语义指标（新增）
        semantic_metrics = self._calculate_semantic_metrics(df_feat, pred_vol)

        # 10. 构建结果
        result = {
            'ts_code': ts_code,
            'prediction_type': 'next_day_volatility',
            'prediction_target': 'Yang-Zhang volatility',
            'predicted_volatility': float(pred_vol),
            'volatility_pct': f"{pred_vol * 100:.2f}%",
            'percentile': semantic_metrics.get('percentile_global', 50),  # 兼容旧接口
            'historical_stats': historical_stats,
            'volatility_trend': volatility_trend,
            'shap_analysis': shap_result,
            'model_metrics': metrics,
            'semantic_metrics': semantic_metrics,  # 新增语义指标
            'data_period': f"最近{days}天",
            'prediction_date': datetime.now().isoformat(),
        }

        return result

    def print_result(self, result: dict):
        """打印预测结果"""
        print(f"\n预测结果: {result['ts_code']}")
        print(f"{'=' * 60}")
        print(f"预测波动率: {result['predicted_volatility']:.6f} ({result['volatility_pct']})")
        print(f"历史分位: {result['percentile']}%")
        
        # 打印语义指标
        semantic = result.get('semantic_metrics', {})
        if semantic:
            print(f"\n【语义指标】")
            
            # 滚动分位数
            rolling_pct = semantic.get('percentile_rolling', {})
            if rolling_pct:
                print(f"滚动分位数:")
                print(f"  近1月: {rolling_pct.get('1m', 'N/A')}%")
                print(f"  近3月: {rolling_pct.get('3m', 'N/A')}%")
                print(f"  近6月: {rolling_pct.get('6m', 'N/A')}%")
                print(f"  近1年: {rolling_pct.get('1y', 'N/A')}%")
            
            # 偏离度
            deviation = semantic.get('deviation_from_mean', {})
            if deviation:
                print(f"相对均值偏离:")
                print(f"  vs 近3月均值: {deviation.get('3m', 'N/A')}%")
                print(f"  vs 近6月均值: {deviation.get('6m', 'N/A')}%")
            
            # 波动状态
            cluster = semantic.get('cluster', {})
            if cluster:
                print(f"波动状态: {cluster.get('regime_desc', 'N/A')}")
            
            # 风险等级
            print(f"综合风险等级: {semantic.get('risk_level_desc', 'N/A')}")
            print(f"风险信号: {semantic.get('risk_signals', [])}")

        if result.get('volatility_trend'):
            trend = result['volatility_trend']
            print(f"\n波动率趋势: {trend['change_pct']:.2f}% vs 前一日")

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
