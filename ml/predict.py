"""指数风险度预测"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from .feature_engineering import IndexRiskFeatureEngineering


class IndexRiskPredictor:
    """指数风险度预测器 (支持多预测窗口)"""

    def __init__(self, model_dir: str = "ml/models"):
        """
        初始化预测器

        Args:
            model_dir: 模型目录路径
        """
        self.model_dir = Path(model_dir)
        self.models = {}  # {horizon: model_bundle}
        self.feature_cols = None
        self.fe = IndexRiskFeatureEngineering()
        self._load_models()

    def _load_models(self):
        """加载训练好的模型"""
        # 查找所有模型文件
        model_files = list(self.model_dir.glob("index_risk_model_*d.pkl"))
        
        if not model_files:
            raise FileNotFoundError(f"未找到模型文件: {self.model_dir}")
        
        for model_file in model_files:
            # 从文件名提取 horizon
            horizon = int(model_file.stem.split('_')[-1].replace('d', ''))
            
            with open(model_file, 'rb') as f:
                data = pickle.load(f)

            model_obj = data['model']
            # 兼容旧模型格式（仅LightGBM）
            if isinstance(model_obj, dict):
                self.models[horizon] = model_obj
            else:
                self.models[horizon] = {
                    'lgb_model': model_obj,
                    'ridge_model': None,
                    'blend_weight_lgb': 1.0,
                }
            
            if self.feature_cols is None:
                self.feature_cols = data['feature_cols']
        
        print(f"模型加载成功: {list(self.models.keys())} 日预测模型")

    def _get_index_data(self, ts_code: str, days: int = 500):
        """
        获取指数日线数据

        Args:
            ts_code: 指数代码
            days: 获取天数

        Returns:
            DataFrame: 原始数据
        """
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

    def predict(self, ts_code: str, days: int = 500, horizons: list = None):
        """
        预测指数未来风险度 (已实现波动率)

        Args:
            ts_code: 指数代码
            days: 获取历史数据天数
            horizons: 预测窗口列表，默认使用所有已加载模型

        Returns:
            dict: 预测结果
        """
        if horizons is None:
            horizons = sorted(self.models.keys())
        
        # 验证 horizons
        for h in horizons:
            if h not in self.models:
                raise ValueError(f"未加载 {h}日预测模型")

        print(f"\n{'=' * 60}")
        print(f"开始预测: {ts_code}")
        print(f"预测窗口: {[f'{h}日' for h in horizons]}")
        print(f"{'=' * 60}")

        # 1. 获取数据
        df = self._get_index_data(ts_code, days)

        # 2. 特征工程
        print("\n特征工程...")
        df_feat = self.fe.create_features(df, target_horizons=horizons)

        # 3. 准备预测数据
        df_valid = df_feat.dropna(subset=self.feature_cols)

        if len(df_valid) == 0:
            raise ValueError("没有有效的预测数据")

        X = df_valid[self.feature_cols]
        print(f"有效预测样本: {len(X)}")

        # 4. 预测每个时间窗口
        predictions_by_horizon = {}
        for horizon in horizons:
            bundle = self.models[horizon]
            lgb_model = bundle['lgb_model']
            ridge_model = bundle.get('ridge_model')
            blend_weight = float(bundle.get('blend_weight_lgb', 1.0))

            pred_lgb = np.exp(lgb_model.predict(X))
            if ridge_model is not None:
                pred_ridge = np.exp(ridge_model.predict(X))
                pred = blend_weight * pred_lgb + (1.0 - blend_weight) * pred_ridge
            else:
                pred = pred_lgb

            predictions_by_horizon[horizon] = pred

        # 5. 计算模型表现
        metrics_by_horizon = {}
        actual_vs_pred = {}
        
        for horizon in horizons:
            target_col = f'rv_{horizon}_fut'
            df_with_target = df_valid.dropna(subset=[target_col])

            if len(df_with_target) > 0:
                common_index = df_with_target.index.intersection(X.index)
                y_true = df_with_target.loc[common_index, target_col].values
                y_pred = predictions_by_horizon[horizon][X.index.get_indexer(common_index)]

                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                
                metrics_by_horizon[horizon] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2)
                }
                
                actual_vs_pred[horizon] = list(zip(
                    df_with_target.loc[common_index, 'trade_date'].astype(str),
                    y_true.tolist(),
                    y_pred.tolist()
                ))
            else:
                metrics_by_horizon[horizon] = {'mae': None, 'rmse': None, 'r2': None}
                actual_vs_pred[horizon] = []

        # 6. 计算分位数语义
        percentiles_by_horizon = {}
        historical_stats_by_horizon = {}
        
        for horizon in horizons:
            target_col = f'rv_{horizon}_fut'
            
            # 使用实际历史波动率计算分位数
            historical_rv = df_valid[target_col].dropna().values
            
            if len(historical_rv) > 0 and horizon in predictions_by_horizon:
                latest_pred = predictions_by_horizon[horizon][-1]
                
                # 计算分位数 (预测值在历史分布中的位置)
                percentile = (np.sum(historical_rv < latest_pred) / len(historical_rv)) * 100
                
                percentiles_by_horizon[horizon] = {
                    'percentile': round(percentile, 1),
                    'label': self._get_percentile_label(percentile),
                    'description': self._get_percentile_description(percentile)
                }
                
                historical_stats_by_horizon[horizon] = {
                    'mean': float(np.mean(historical_rv)),
                    'std': float(np.std(historical_rv)),
                    'min': float(np.min(historical_rv)),
                    'max': float(np.max(historical_rv)),
                    'q25': float(np.percentile(historical_rv, 25)),
                    'q50': float(np.percentile(historical_rv, 50)),
                    'q75': float(np.percentile(historical_rv, 75)),
                }

        # 7. 构建结果
        result = {
            'ts_code': ts_code,
            'data_period': f"最近{days}天",
            'target_horizons': horizons,
            'total_records': len(df),
            'valid_predictions': len(df_valid),
            'latest_predictions': {
                h: float(predictions_by_horizon[h][-1]) 
                for h in horizons
            },
            'percentiles': percentiles_by_horizon,
            'historical_stats': historical_stats_by_horizon,
            'metrics': metrics_by_horizon,
            'actual_vs_predicted': actual_vs_pred,
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

        print(f"\n数据概况:")
        print(f"  时间范围: {result['data_period']}")
        print(f"  总记录数: {result['total_records']}")
        print(f"  有效预测数: {result['valid_predictions']}")

        print(f"\n最新预测:")
        for horizon in result['target_horizons']:
            if horizon in result['latest_predictions']:
                rv = result['latest_predictions'][horizon]
                period = "一周" if horizon == 5 else "一月"
                
                print(f"\n  【{horizon}日 ({period})】")
                print(f"    已实现波动率: {rv:.6f}")
                
                # 分位数语义
                if horizon in result.get('percentiles', {}):
                    p = result['percentiles'][horizon]
                    print(f"    历史分位数: {p['percentile']}% ({p['label']})")
                    print(f"    语义解读: {p['description']}")
                
                # 历史统计
                if horizon in result.get('historical_stats', {}):
                    stats = result['historical_stats'][horizon]
                    print(f"    历史范围: [{stats['min']:.6f}, {stats['max']:.6f}]")
                    print(f"    历史中位数: {stats['q50']:.6f}")

        print(f"\n模型性能:")
        for horizon in result['target_horizons']:
            if horizon in result['metrics']:
                metrics = result['metrics'][horizon]
                period = "一周" if horizon == 5 else "一月"
                if metrics['mae'] is not None:
                    print(f"  {horizon}日 ({period}): MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.4f}")
                else:
                    print(f"  {horizon}日 ({period}): (无评估数据)")

        return result


def predict_index_risk(ts_code: str, days: int = 500, model_dir: str = None, horizons: list = None):
    """
    便捷函数: 预测指数风险度

    Args:
        ts_code: 指数代码
        days: 获取历史数据天数
        model_dir: 模型目录路径
        horizons: 预测窗口列表

    Returns:
        dict: 预测结果
    """
    if model_dir is None:
        model_dir = "ml/models"

    predictor = IndexRiskPredictor(model_dir)
    result = predictor.predict(ts_code, days, horizons)
    predictor.print_result(result)

    return result


def predict_multi_indices(ts_codes: list, days: int = 500, model_dir: str = None, horizons: list = None):
    """
    便捷函数: 批量预测多个指数

    Args:
        ts_codes: 指数代码列表
        days: 获取历史数据天数
        model_dir: 模型目录路径
        horizons: 预测窗口列表

    Returns:
        list: 预测结果列表
    """
    if model_dir is None:
        model_dir = "ml/models"

    predictor = IndexRiskPredictor(model_dir)
    results = []

    for i, code in enumerate(ts_codes, 1):
        print(f"\n\n进度: {i}/{len(ts_codes)}")
        try:
            result = predictor.predict(code, days, horizons)
            results.append(result)
        except Exception as e:
            print(f"预测 {code} 失败: {e}")
            results.append({'ts_code': code, 'error': str(e)})

    # 汇总
    print(f"\n\n{'=' * 60}")
    print("批量预测完成!")
    print(f"{'=' * 60}")

    success = sum(1 for r in results if 'latest_predictions' in r)
    print(f"成功: {success}/{len(ts_codes)}")

    return results


if __name__ == "__main__":
    # 示例: 预测上证指数风险度
    result = predict_index_risk(
        ts_code="000001.SH",
        days=500,
        horizons=[5, 20]  # 一周和一月
    )
