"""
模型对比分析
所有模型预测同一目标：Yang-Zhang 波动率
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

ml_dir = Path(__file__).parent

print("=" * 70)
print("模型对比分析")
print("所有模型预测同一目标：Yang-Zhang 波动率")
print("=" * 70)

# 1. 加载数据
df = pd.read_csv(ml_dir / 'dataset' / 'index_features.csv')
yz_vol = df['yang_zhang_vol'].values
true_vol = df['target_vol'].values

# 2. 运行基线模型
try:
    from .baseline_models import BaselineComparator, ModelMetrics
except ImportError:
    from baseline_models import BaselineComparator, ModelMetrics

print("\n【基线模型】")
comparator = BaselineComparator()
baseline_results = comparator.compare_all(yz_vol, true_vol, min_train=20)

# 3. 加载 LightGBM 模型
print("\n【LightGBM】")
model_path = ml_dir / "models" / "volatility_model_lgb.pkl"
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

lgb_model = model_data['model']
feature_cols = model_data['feature_cols']

# 检查缺失特征
missing_features = [c for c in feature_cols if c not in df.columns]
if missing_features:
    print(f"添加 {len(missing_features)} 个 GARCH 特征...")
    try:
        from .garch_features import add_garch_features_to_df
    except ImportError:
        from garch_features import add_garch_features_to_df
    df = add_garch_features_to_df(df, min_train_size=500)

# 准备数据
valid_features = [c for c in feature_cols if c in df.columns]
df_clean = df.dropna(subset=['target_vol'] + valid_features)
X = df_clean[valid_features]
y_true = df_clean['target_vol'].values

# 预测
y_pred = np.exp(lgb_model.predict(X[valid_features]))
lgb_metrics = ModelMetrics.calculate_all(y_true, y_pred)
lgb_metrics['model'] = 'LightGBM'

print(f"  MAE: {lgb_metrics['mae']:.6f}")
print(f"  RMSE: {lgb_metrics['rmse']:.6f}")
print(f"  R²: {lgb_metrics['r2']:.4f}")
print(f"  MAPE: {lgb_metrics['mape']:.2f}%")
print(f"  方向准确率: {lgb_metrics['direction_accuracy']:.1f}%")
print(f"  命中率: {lgb_metrics['hit_rate']:.1f}%")

# 4. 合并结果
all_results = pd.DataFrame([lgb_metrics] + baseline_results.to_dict('records'))
all_results = all_results.sort_values('mae')

print("\n" + "=" * 70)
print("完整对比表（按 MAE 排序）")
print("=" * 70)
display_cols = ['model', 'mae', 'rmse', 'r2', 'mape', 'direction_accuracy', 'hit_rate']
print(all_results[display_cols].to_string(index=False))

# 5. 保存
all_results.to_csv(ml_dir / 'models' / 'model_comparison_yz.csv', index=False)
print(f"\n已保存到: ml/models/model_comparison_yz.csv")
