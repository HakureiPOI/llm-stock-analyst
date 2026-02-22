import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import pickle
import json
from datetime import datetime


class VolatilityModel:
    """波动率预测模型"""

    def __init__(self, model_path: str = None):
        """
        初始化模型

        Args:
            model_path: 模型保存路径
        """
        self.model = None
        self.model_path = model_path or "ml/models/volatility_model.pkl"
        self.metadata_path = str(Path(self.model_path).parent / "model_metadata.json")

    def prepare_data(self, df: pd.DataFrame, train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> tuple:
        """
        准备训练数据并分割

        Args:
            df: 特征工程后的数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        print("准备数据...")

        # 目标变量
        target_column = 'future_volatility'

        # 删除不需要的列
        columns_to_drop = [
            'ts_code', 'trade_date', 'log_return', target_column
        ]

        X = df.drop(columns=columns_to_drop)
        y = df[target_column]

        # 按时间排序
        df_sorted = df.sort_values(by='trade_date')
        X = X.reindex(df_sorted.index)
        y = y.reindex(df_sorted.index)

        # 分割数据 (训练/验证/测试 = 70%/15%/15%)
        total_samples = len(df_sorted)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)

        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]

        X_val = X.iloc[train_size:train_size + val_size]
        y_val = y.iloc[train_size:train_size + val_size]

        X_test = X.iloc[train_size + val_size:]
        y_test = y.iloc[train_size + val_size:]

        print(f"总样本: {total_samples}")
        print(f"训练集: {len(X_train)}")
        print(f"验证集: {len(X_val)}")
        print(f"测试集: {len(X_test)}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(self, X_train, y_train, X_val, y_val,
             random_state: int = 42) -> lgb.LGBMRegressor:
        """
        训练LightGBM模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            random_state: 随机种子

        Returns:
            训练好的模型
        """
        print("\n训练LightGBM模型...")

        # 初始化模型
        self.model = lgb.LGBMRegressor(
            random_state=random_state,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=-1,
            num_leaves=31,
            verbose=-1
        )

        # 训练
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        print("训练完成!")

        # 验证集评估
        y_pred_val = self.model.predict(X_val)
        self._print_metrics(y_val, y_pred_val, "验证集")

        return self.model

    def _print_metrics(self, y_true, y_pred, dataset_name: str):
        """打印评估指标"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print(f"\n{dataset_name}指标:")
        print(f"  MAE:  {mae:.6f}")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²:   {r2:.6f}")

        return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

    def evaluate(self, X_test, y_test) -> dict:
        """
        评估模型

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            评估指标字典
        """
        print("\n评估测试集...")

        y_pred = self.model.predict(X_test)
        metrics = self._print_metrics(y_test, y_pred, "测试集")

        return metrics

    def save_model(self, metadata: dict = None):
        """
        保存模型和元数据

        Args:
            metadata: 元数据字典
        """
        # 确保目录存在
        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\n模型已保存: {self.model_path}")

        # 构建元数据
        if metadata is None:
            metadata = {}

        default_metadata = {
            'model_type': 'LightGBM',
            'task': 'volatility_prediction',
            'target': 'future_volatility (5-day log return std)',
            'training_date': datetime.now().isoformat(),
            'saved_at': datetime.now().isoformat(),
            'model_path': str(self.model_path)
        }

        # 合并元数据
        final_metadata = {**default_metadata, **metadata}

        # 保存元数据
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(final_metadata, f, indent=2, ensure_ascii=False)
        print(f"元数据已保存: {self.metadata_path}")

        return final_metadata

    def load_model(self):
        """加载模型和元数据"""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"模型已加载: {self.model_path}")

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"元数据已加载")

        return metadata

    def predict(self, X) -> np.ndarray:
        """
        预测

        Args:
            X: 特征数据

        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练或加载")
        return self.model.predict(X)

    def run_pipeline(self, input_file: str, model_save_path: str = None):
        """
        完整训练流程

        Args:
            input_file: 特征数据文件
            model_save_path: 模型保存路径

        Returns:
            (模型, 元数据, 测试集指标)
        """
        # 加载数据
        print(f"加载数据: {input_file}")
        df = pd.read_csv(input_file)

        # 设置模型保存路径
        if model_save_path:
            self.model_path = model_save_path
            self.metadata_path = str(Path(model_save_path).parent / "model_metadata.json")

        # 准备数据
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(df)

        # 训练模型
        model = self.train(X_train, y_train, X_val, y_val)

        # 评估模型
        test_metrics = self.evaluate(X_test, y_test)

        # 保存模型和元数据
        metadata = {
            'input_file': input_file,
            'total_samples': len(df),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_count': X_train.shape[1],
            'test_metrics': test_metrics
        }

        final_metadata = self.save_model(metadata)

        return model, final_metadata, test_metrics


if __name__ == "__main__":
    trainer = VolatilityModel()

    # 训练并保存模型
    model, metadata, metrics = trainer.run_pipeline(
        input_file="/home/hakurei/llm-stock-analyst/ml/dataset/kline_dataset_features.csv",
        model_save_path="/home/hakurei/llm-stock-analyst/ml/models/volatility_model.pkl"
    )

    print("\n" + "="*50)
    print("训练完成!")
    print("="*50)
