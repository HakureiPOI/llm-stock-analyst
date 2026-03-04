"""
波动率预测模型训练 V2
主模型：LightGBM + 基础特征 + OHLC区间特征 + GARCH特征
基线模型：EWMA, GARCH, EGARCH, GJR-GARCH
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .baseline_models import ModelMetrics, BaselineComparator


class VolatilityModel:
    """波动率预测模型"""

    def __init__(self, model_dir: str = None):
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = Path(__file__).parent / "models"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_cols = None
        self.feature_importance = None

    def prepare_data(self, df: pd.DataFrame,
                     feature_cols: List[str] = None,
                     target_col: str = 'target_vol',
                     min_train: int = 500) -> Tuple:
        """
        准备训练数据

        Args:
            df: 特征工程后的数据
            feature_cols: 特征列名 (None则自动选择)
            target_col: 目标列名
            min_train: 最小训练样本起始点

        Returns:
            (df_clean, feature_cols)
        """
        df = df.copy()

        # 自动选择特征列
        if feature_cols is None:
            drop_cols = ['ts_code', 'trade_date', 'target_vol', 'target_vol_log',
                        'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
            feature_cols = [col for col in df.columns if col not in drop_cols]

        # 只使用有效的特征列（过滤全为NaN的特征）
        valid_features = []
        for col in feature_cols:
            if col in df.columns and df[col].notna().sum() > min_train:
                valid_features.append(col)
            elif col in df.columns:
                print(f"  跳过特征 '{col}' (有效值: {df[col].notna().sum()})")

        feature_cols = valid_features

        # 删除NaN
        df_clean = df.dropna(subset=[target_col] + feature_cols)

        print(f"有效样本: {len(df_clean)}")
        print(f"特征数量: {len(feature_cols)}")

        return df_clean, feature_cols

    def walk_forward_train(self, df: pd.DataFrame,
                           feature_cols: List[str] = None,
                           target_col: str = 'target_vol',
                           train_window: int = 500,
                           test_window: int = 63,
                           valid_ratio: float = 0.2) -> dict:
        """
        Walk-Forward 训练验证

        Args:
            df: 特征工程后的数据
            feature_cols: 特征列名
            target_col: 目标列名
            train_window: 训练窗口大小
            test_window: 测试窗口大小
            valid_ratio: 验证集比例

        Returns:
            训练结果字典
        """
        # 排序
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 准备数据
        df_clean, feature_cols = self.prepare_data(df, feature_cols, target_col, min_train=train_window)
        self.feature_cols = feature_cols

        all_metrics = []
        all_predictions = []

        fold = 0
        for start in range(train_window, len(df_clean), test_window):
            fold += 1

            # 划分数据
            train_full = df_clean.iloc[start - train_window:start]
            test = df_clean.iloc[start:start + test_window]

            if len(test) == 0:
                break

            # 训练集/验证集划分
            n_train = len(train_full)
            n_valid = int(n_train * valid_ratio)
            train = train_full.iloc[:-n_valid]
            valid = train_full.iloc[-n_valid:]

            # 特征和目标
            X_train = train[feature_cols]
            y_train = np.log(train[target_col])  # 对数变换
            X_valid = valid[feature_cols]
            y_valid = np.log(valid[target_col])
            X_test = test[feature_cols]
            y_test = test[target_col]

            # 训练模型
            model = lgb.LGBMRegressor(
                n_estimators=5000,
                learning_rate=0.02,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(200, verbose=False)]
            )

            # 预测
            y_pred = np.exp(model.predict(X_test))  # 反变换

            # 计算指标
            metrics = ModelMetrics.calculate_all(y_test.values, y_pred)
            all_metrics.append(metrics)

            # 保存预测结果
            pred_df = pd.DataFrame({
                'trade_date': test['trade_date'].values,
                'y_true': y_test.values,
                'y_pred': y_pred
            })
            all_predictions.append(pred_df)

            if fold <= 3:
                print(f"  Fold {fold}: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.4f}")

        # 合并预测结果
        predictions_df = pd.concat(all_predictions).sort_values('trade_date').reset_index(drop=True)

        # 最终模型: 用全部数据重新训练
        print(f"\n训练最终模型...")
        n_valid = int(len(df_clean) * valid_ratio)
        train_final = df_clean.iloc[:-n_valid]
        valid_final = df_clean.iloc[-n_valid:]

        X_train_final = train_final[feature_cols]
        y_train_final = np.log(train_final[target_col])
        X_valid_final = valid_final[feature_cols]
        y_valid_final = np.log(valid_final[target_col])

        final_model = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        final_model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_valid_final, y_valid_final)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )

        self.model = final_model

        # 特征重要性
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        self.feature_importance = importance_df

        # 汇总指标
        avg_metrics = {
            key: float(np.mean([m[key] for m in all_metrics]))
            for key in all_metrics[0].keys()
        }

        results = {
            'avg_metrics': avg_metrics,
            'fold_metrics': all_metrics,
            'folds': fold,
            'predictions': predictions_df,
            'feature_importance': importance_df.to_dict('records')
        }

        print(f"\n训练完成!")
        print(f"平均指标: MAE={avg_metrics['mae']:.6f}, RMSE={avg_metrics['rmse']:.6f}, R²={avg_metrics['r2']:.4f}")

        # 打印Top10特征
        print(f"\nTop 10 特征重要性:")
        for _, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.2f}")

        return results

    def save_model(self, results: dict = None):
        """保存模型"""
        model_path = self.model_dir / "volatility_model_lgb.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_cols': self.feature_cols,
                'feature_importance': self.feature_importance
            }, f)
        print(f"模型已保存: {model_path}")

        # 保存特征重要性
        if self.feature_importance is not None:
            importance_path = self.model_dir / "feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
            print(f"特征重要性已保存: {importance_path}")

        # 保存元数据
        metadata = {
            'model_type': 'LightGBM',
            'task': 'next_day_volatility_prediction',
            'target': 'Yang-Zhang volatility (t+1)',
            'training_date': datetime.now().isoformat(),
            'feature_count': len(self.feature_cols) if self.feature_cols else 0,
        }

        if results and 'avg_metrics' in results:
            metadata['metrics'] = results['avg_metrics']

        metadata_path = self.model_dir / "model_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"元数据已保存: {metadata_path}")


def run_full_training(data_file: str = None,
                      include_garch: bool = True,
                      compare_baselines: bool = True) -> dict:
    """
    完整训练流程

    Args:
        data_file: 特征数据文件路径
        include_garch: 是否包含GARCH特征
        compare_baselines: 是否比较基线模型

    Returns:
        训练结果
    """
    ml_dir = Path(__file__).parent

    # 读取数据
    if data_file is None:
        data_file = ml_dir / "dataset" / "index_features.csv"

    print(f"加载数据: {data_file}")
    df = pd.read_csv(data_file)

    # 添加GARCH特征
    if include_garch:
        try:
            from .garch_features import add_garch_features_to_df
            print("\n添加GARCH特征...")
            df = add_garch_features_to_df(df, min_train_size=500)
        except Exception as e:
            print(f"GARCH特征添加失败: {e}")

    # 训练主模型
    print("\n" + "=" * 60)
    print("训练主模型 (LightGBM)")
    print("=" * 60)

    model = VolatilityModel()
    results = model.walk_forward_train(df)

    # 保存模型
    model.save_model(results)

    # 比较基线模型
    if compare_baselines:
        print("\n" + "=" * 60)
        print("比较基线模型")
        print("=" * 60)

        comparator = BaselineComparator()
        returns = df['log_ret'].values
        true_vol = df['target_vol'].values

        baseline_results = comparator.compare_all(returns, true_vol, min_train=200)
        comparator.print_comparison(baseline_results)

        # 保存基线比较结果
        baseline_results.to_csv(ml_dir / "models" / "baseline_comparison.csv", index=False)

    # 打印最终总结
    print("\n" + "=" * 70)
    print("训练完成! 模型性能总结")
    print("=" * 70)

    metrics = results['avg_metrics']
    print(f"\n【LightGBM 主模型】")
    print(f"  MAE:           {metrics['mae']:.6f}")
    print(f"  RMSE:          {metrics['rmse']:.6f}")
    print(f"  R²:            {metrics['r2']:.4f}")
    print(f"  MAPE:          {metrics['mape']:.2f}%")
    print(f"  方向准确率:     {metrics['direction_accuracy']:.1f}%")
    print(f"  相关系数:      {metrics['correlation']:.4f}")
    print(f"  命中率(±20%):  {metrics['hit_rate']:.1f}%")

    return results


if __name__ == "__main__":
    results = run_full_training(
        data_file=None,
        include_garch=True,
        compare_baselines=True
    )
