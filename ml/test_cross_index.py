"""
跨指数预测能力测试
测试在上证指数上训练的模型在其他指数上的表现
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.predict import VolatilityPredictor
from ml.feature_engineering import VolatilityFeatureEngineering
from ml.baseline_models import ModelMetrics
from tsdata.index import index_data


def get_index_data_for_test(ts_code: str, start_date: str = None, end_date: str = None, days: int = 2500) -> pd.DataFrame:
    """获取指数日线数据（默认获取约10年数据）"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    print(f"获取 {ts_code} 数据: {start_date} 至 {end_date}")
    df = index_data.get_index_daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )

    if df is None or df.empty:
        raise ValueError(f"未获取到 {ts_code} 的数据")

    print(f"获取到 {len(df)} 条记录")
    return df


def evaluate_on_index(predictor: VolatilityPredictor, ts_code: str, index_name: str, days: int = 2500) -> dict:
    """在指定指数上评估模型"""
    print(f"\n{'=' * 60}")
    print(f"测试指数: {index_name} ({ts_code})")
    print(f"{'=' * 60}")

    try:
        # 获取数据
        df = get_index_data_for_test(ts_code, days=days)

        # 特征工程
        fe = VolatilityFeatureEngineering(yz_window=20)
        df_feat = fe.create_features(df, include_garch_features=False)

        # 添加GARCH特征
        try:
            from ml.garch_features import add_garch_features_to_df
            print("添加GARCH特征...")
            df_feat = add_garch_features_to_df(df_feat, min_train_size=100)
        except Exception as e:
            print(f"GARCH特征添加失败: {e}")

        # 准备特征 - 对齐模型的特征列表
        available_features = [col for col in predictor.feature_cols if col in df_feat.columns]
        missing_features = [col for col in predictor.feature_cols if col not in df_feat.columns]

        if missing_features:
            print(f"缺失特征数: {len(missing_features)}")
            # 为缺失特征填充默认值
            for col in missing_features:
                df_feat[col] = 0.0

        df_valid = df_feat.dropna(subset=['target_vol'] + available_features)
        print(f"有效数据量: {len(df_valid)}")

        if len(df_valid) < 200:
            return {"ts_code": ts_code, "index_name": index_name, "error": f"数据不足({len(df_valid)}条)"}

        # Walk-Forward 评估
        train_window = min(500, len(df_valid) // 2)  # 自适应训练窗口
        test_window = 63
        all_metrics = []
        all_preds = []

        # 使用模型的完整特征列表
        for start in range(train_window, len(df_valid), test_window):
            test = df_valid.iloc[start:start + test_window]
            if len(test) == 0:
                break

            X_test = test[predictor.feature_cols]  # 使用完整特征列表
            y_true = test['target_vol'].values
            y_pred = np.exp(predictor.model.predict(X_test))

            metrics = ModelMetrics.calculate_all(y_true, y_pred)
            all_metrics.append(metrics)
            all_preds.append(pd.DataFrame({
                'trade_date': test['trade_date'].values,
                'y_true': y_true,
                'y_pred': y_pred
            }))

        if len(all_metrics) == 0:
            return {"ts_code": ts_code, "index_name": index_name, "error": "没有有效的测试折"}

        # 汇总
        avg_metrics = {
            key: float(np.mean([m[key] for m in all_metrics]))
            for key in all_metrics[0].keys()
        }

        # 合并预测
        predictions_df = pd.concat(all_preds)

        return {
            "ts_code": ts_code,
            "index_name": index_name,
            "data_count": len(df_valid),
            "fold_count": len(all_metrics),
            "metrics": avg_metrics,
            "predictions": predictions_df
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ts_code": ts_code, "index_name": index_name, "error": str(e)}


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("跨指数波动率预测能力测试")
    print("=" * 70)
    print("\n训练数据: 上证指数 (000001.SH)")
    print("测试数据: 其他主要市场指数")

    # 加载模型
    predictor = VolatilityPredictor()

    # 测试指数列表
    test_indices = [
        ("000001.SH", "上证指数"),      # 训练集（对照组）
        ("399001.SZ", "深证成指"),      # 深圳市场
        ("399006.SZ", "创业板指"),      # 创业板
        ("000300.SH", "沪深300"),       # 大盘蓝筹
        ("000016.SH", "上证50"),        # 超大盘
        ("000905.SH", "中证500"),       # 中盘
        ("000852.SH", "中证1000"),      # 小盘
    ]

    results = []

    for ts_code, index_name in test_indices:
        result = evaluate_on_index(predictor, ts_code, index_name)
        results.append(result)

    # 汇总报告
    print("\n" + "=" * 70)
    print("跨指数预测结果汇总")
    print("=" * 70)

    print(f"\n{'指数':<15} {'数据量':>8} {'MAE':>12} {'RMSE':>12} {'R²':>10} {'MAPE':>10} {'命中率':>10}")
    print("-" * 70)

    summary_data = []
    for r in results:
        if 'error' in r:
            print(f"{r['index_name']:<15} 错误: {r['error']}")
            continue

        m = r['metrics']
        print(f"{r['index_name']:<15} {r['data_count']:>8} {m['mae']:>12.6f} {m['rmse']:>12.6f} {m['r2']:>10.4f} {m['mape']:>9.2f}% {m['hit_rate']:>9.2f}%")

        summary_data.append({
            'ts_code': r['ts_code'],
            'index_name': r['index_name'],
            'data_count': r['data_count'],
            'fold_count': r['fold_count'],
            'mae': m['mae'],
            'rmse': m['rmse'],
            'r2': m['r2'],
            'mape': m['mape'],
            'direction_accuracy': m['direction_accuracy'],
            'correlation': m['correlation'],
            'hit_rate': m['hit_rate']
        })

    # 保存结果
    ml_dir = Path(__file__).parent
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(ml_dir / "models" / "cross_index_evaluation.csv", index=False)
    print(f"\n结果已保存: {ml_dir / 'models' / 'cross_index_evaluation.csv'}")

    # 分析
    print("\n" + "=" * 70)
    print("分析结论")
    print("=" * 70)

    if len(summary_data) > 1:
        # 对比训练集与跨指数表现
        train_mae = summary_data[0]['mae']
        cross_mae = np.mean([d['mae'] for d in summary_data[1:]])
        train_r2 = summary_data[0]['r2']
        cross_r2 = np.mean([d['r2'] for d in summary_data[1:]])

        print(f"\n训练集 (上证指数):")
        print(f"  MAE = {train_mae:.6f}")
        print(f"  R²  = {train_r2:.4f}")

        print(f"\n跨指数平均:")
        print(f"  MAE = {cross_mae:.6f} (相对训练集: {cross_mae/train_mae:.2f}x)")
        print(f"  R²  = {cross_r2:.4f} (相对训练集: {cross_r2/train_r2:.2f}x)")

        if cross_mae < train_mae * 1.5:
            print(f"\n结论: 模型跨指数泛化能力良好 (MAE增量 < 50%)")
        elif cross_mae < train_mae * 2.0:
            print(f"\n结论: 模型跨指数泛化能力一般 (MAE增量 50%-100%)")
        else:
            print(f"\n结论: 模型跨指数泛化能力较弱 (MAE增量 > 100%)")

    return results


if __name__ == "__main__":
    results = main()
