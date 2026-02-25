"""指数风险度模型训练 - Walk-Forward 验证"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import pickle
import json
from datetime import datetime


class IndexRiskModel:
    """指数风险度预测模型 (支持多预测窗口)"""

    def __init__(self, model_path: str = None):
        """
        初始化模型

        Args:
            model_path: 模型保存路径 (不含扩展名)
        """
        self.models = {}  # 存储 {horizon: model}
        self.feature_cols = None
        self.target_horizons = [5, 20]
        
        # 模型保存路径
        if model_path:
            self.model_dir = Path(model_path).parent
        else:
            self.model_dir = Path("ml/models")
        
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, df: pd.DataFrame, target_horizons: list = [5, 20]) -> tuple:
        """
        准备训练数据

        Args:
            df: 特征工程后的数据
            target_horizons: 目标预测窗口列表

        Returns:
            (df_clean, feature_cols, target_cols)
        """
        self.target_horizons = target_horizons
        target_cols = [f'rv_{h}_fut' for h in target_horizons]

        # 删除不需要的列
        drop_cols = ['ts_code', 'trade_date'] + target_cols
        self.feature_cols = [col for col in df.columns if col not in drop_cols]

        # 删除 NaN
        df_clean = df.dropna(subset=target_cols + self.feature_cols)

        print(f"有效样本: {len(df_clean)}")
        print(f"特征数量: {len(self.feature_cols)}")
        print(f"预测目标: {[f'{h}日RV' for h in target_horizons]}")

        return df_clean, self.feature_cols, target_cols

    def walk_forward_train(self, df: pd.DataFrame,
                           train_window: int = 756,
                           test_window: int = 63,
                           valid_ratio: float = 0.2,
                           target_horizons: list = [5, 20]) -> dict:
        """
        Walk-Forward 训练验证 (支持多预测窗口)

        Args:
            df: 特征工程后的数据 (需按时间排序)
            train_window: 训练窗口大小
            test_window: 测试窗口大小
            valid_ratio: 验证集比例
            target_horizons: 目标预测窗口列表

        Returns:
            dict: 训练结果
        """
        # 准备数据
        df = df.sort_values('trade_date').reset_index(drop=True)
        df_clean, feature_cols, target_cols = self.prepare_data(df, target_horizons)

        results = {}
        
        for horizon, target_col in zip(target_horizons, target_cols):
            print(f"\n{'='*60}")
            print(f"训练 {horizon}日预测模型")
            print(f"{'='*60}")
            
            maes, rmses = [], []
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
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                maes.append(mae)
                rmses.append(rmse)

                # 保存预测结果
                pred_df = pd.DataFrame({
                    'trade_date': test['trade_date'].values,
                    'y_true': y_test.values,
                    'y_pred': y_pred
                })
                all_predictions.append(pred_df)

                if fold <= 3:  # 只显示前3个fold
                    print(f"  Fold {fold}: MAE={mae:.6f}, RMSE={rmse:.6f}")

            # 合并预测结果
            predictions_df = pd.concat(all_predictions).sort_values('trade_date').reset_index(drop=True)

            # 最终模型: 用全部数据重新训练
            print(f"\n用全部数据训练 {horizon}日 最终模型...")
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

            # 保存模型
            self.models[horizon] = final_model

            # 汇总结果
            results[horizon] = {
                'avg_mae': float(np.mean(maes)),
                'avg_rmse': float(np.mean(rmses)),
                'folds': fold,
                'predictions': predictions_df
            }

            print(f"\n{horizon}日模型训练完成!")
            print(f"平均 MAE:  {results[horizon]['avg_mae']:.6f}")
            print(f"平均 RMSE: {results[horizon]['avg_rmse']:.6f}")

        return results

    def save_model(self, metadata: dict = None):
        """保存模型和元数据"""
        import json

        # 保存每个模型
        for horizon, model in self.models.items():
            model_path = self.model_dir / f"index_risk_model_{horizon}d.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'feature_cols': self.feature_cols,
                    'horizon': horizon
                }, f)
            print(f"{horizon}日模型已保存: {model_path}")

        # 保存元数据
        metadata_path = self.model_dir / "model_metadata.json"
        default_metadata = {
            'model_type': 'LightGBM',
            'task': 'index_risk_prediction',
            'target_horizons': list(self.models.keys()),
            'training_date': datetime.now().isoformat(),
        }

        if metadata:
            default_metadata.update(metadata)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(default_metadata, f, indent=2, ensure_ascii=False)
        print(f"元数据已保存: {metadata_path}")

    def run_pipeline(self, input_file: str, model_save_path: str = None,
                     target_horizons: list = [5, 20]) -> dict:
        """
        完整训练流程

        Args:
            input_file: 特征数据文件
            model_save_path: 模型保存路径
            target_horizons: 目标预测窗口列表

        Returns:
            dict: 训练结果
        """
        print(f"加载数据: {input_file}")
        df = pd.read_csv(input_file)

        if model_save_path:
            self.model_dir = Path(model_save_path).parent

        # 训练
        results = self.walk_forward_train(df, target_horizons=target_horizons)

        # 保存
        metadata = {
            'input_file': input_file,
            'total_samples': len(df),
            'feature_count': len(self.feature_cols),
        }
        self.save_model(metadata)

        return results


if __name__ == "__main__":
    ml_dir = Path(__file__).parent
    trainer = IndexRiskModel()

    results = trainer.run_pipeline(
        input_file=str(ml_dir / "dataset" / "index_features.csv"),
        model_save_path=str(ml_dir / "models" / "index_risk_model.pkl"),
        target_horizons=[5, 20]  # 一周和一月
    )

    print("\n" + "=" * 50)
    print("训练完成!")
    print("=" * 50)
    
    for horizon, res in results.items():
        print(f"\n{horizon}日模型: MAE={res['avg_mae']:.6f}, RMSE={res['avg_rmse']:.6f}")
