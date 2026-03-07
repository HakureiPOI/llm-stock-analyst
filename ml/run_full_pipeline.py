"""
完整训练流程
1. 数据获取
2. 特征提取
3. 模型训练
4. 基线对比
"""
import sys
from pathlib import Path

ml_dir = Path(__file__).parent
sys.path.insert(0, str(ml_dir.parent))

print("=" * 70)
print("完整训练流程")
print("=" * 70)

# ==================== 1. 数据获取 ====================
print("\n" + "=" * 70)
print("1. 数据获取")
print("=" * 70)

from datetime import datetime, timedelta
from tsdata.index import index_data

ts_code = "000001.SH"
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 10)  # 10年数据

start_date_str = start_date.strftime("%Y%m%d")
end_date_str = end_date.strftime("%Y%m%d")

print(f"获取 {ts_code} 数据: {start_date_str} 至 {end_date_str}")

df = index_data.get_index_daily(
    ts_code=ts_code,
    start_date=start_date_str,
    end_date=end_date_str
)

if df is None or df.empty:
    raise ValueError("数据获取失败")

df = df.sort_values('trade_date').reset_index(drop=True)
print(f"获取到 {len(df)} 条记录")

# 保存原始数据
raw_data_path = ml_dir / "dataset" / f"index_daily_{ts_code.replace('.', '_')}_10years.csv"
df.to_csv(raw_data_path, index=False, encoding='utf-8-sig')
print(f"原始数据已保存: {raw_data_path}")

# ==================== 2. 特征提取 ====================
print("\n" + "=" * 70)
print("2. 特征提取")
print("=" * 70)

from feature_engineering import VolatilityFeatureEngineering

fe = VolatilityFeatureEngineering(yz_window=20)
df_feat = fe.create_features(df)

# 保存特征数据
feat_data_path = ml_dir / "dataset" / "index_features.csv"
df_feat.to_csv(feat_data_path, index=False, encoding='utf-8-sig')
print(f"特征数据已保存: {feat_data_path}")
print(f"特征数量: {len(fe.get_feature_columns(df_feat))}")

# ==================== 3. 模型训练 ====================
print("\n" + "=" * 70)
print("3. 模型训练")
print("=" * 70)

from train_model import VolatilityModel
from garch_features import add_garch_features_to_df

# 添加 GARCH 特征
print("添加 GARCH 特征...")
df_with_garch = add_garch_features_to_df(df_feat, min_train_size=500)

# 训练模型
model = VolatilityModel()
results = model.walk_forward_train(df_with_garch)

# 保存模型
model.save_model(results)

# ==================== 4. 基线对比 ====================
print("\n" + "=" * 70)
print("4. 基线对比")
print("=" * 70)

import numpy as np
import pandas as pd
from baseline_models import BaselineComparator, ModelMetrics
import pickle

# 准备数据
yz_vol = df_feat['yang_zhang_vol'].values
true_vol = df_feat['target_vol'].values

# 运行基线模型
print("评估基线模型...")
comparator = BaselineComparator()
baseline_results = comparator.compare_all(yz_vol, true_vol, min_train=20)

# 加载训练好的模型
model_path = ml_dir / "models" / "volatility_model_lgb.pkl"
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

lgb_model = model_data['model']
feature_cols = model_data['feature_cols']

# 准备预测数据
valid_features = [c for c in feature_cols if c in df_with_garch.columns]
df_clean = df_with_garch.dropna(subset=['target_vol'] + valid_features)
X = df_clean[valid_features]
y_true = df_clean['target_vol'].values

# LightGBM 预测
y_pred = np.exp(lgb_model.predict(X[valid_features]))
lgb_metrics = ModelMetrics.calculate_all(y_true, y_pred)
lgb_metrics['model'] = 'LightGBM'

# 合并结果
all_results = pd.DataFrame([lgb_metrics] + baseline_results.to_dict('records'))
all_results = all_results.sort_values('mae')

# 打印结果
print("\n" + "=" * 70)
print("模型对比结果")
print("=" * 70)
display_cols = ['model', 'mae', 'rmse', 'r2', 'mape', 'direction_accuracy', 'hit_rate']
print(all_results[display_cols].to_string(index=False))

# 保存对比结果
comparison_path = ml_dir / "models" / "model_comparison_yz.csv"
all_results.to_csv(comparison_path, index=False)
print(f"\n对比结果已保存: {comparison_path}")

# ==================== 5. 总结 ====================
print("\n" + "=" * 70)
print("训练完成！")
print("=" * 70)
print(f"""
输出文件:
  - 原始数据: {raw_data_path}
  - 特征数据: {feat_data_path}
  - 模型文件: {model_path}
  - 对比结果: {comparison_path}

模型性能:
  - MAE: {lgb_metrics['mae']:.6f}
  - RMSE: {lgb_metrics['rmse']:.6f}
  - R²: {lgb_metrics['r2']:.4f}
  - MAPE: {lgb_metrics['mape']:.2f}%
  - 方向准确率: {lgb_metrics['direction_accuracy']:.1f}%
  - 命中率: {lgb_metrics['hit_rate']:.1f}%
""")
