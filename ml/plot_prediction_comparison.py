"""
绘制近1年真实值vs预测值对比图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
from pathlib import Path

from .feature_engineering import VolatilityFeatureEngineering
from .garch_features import add_garch_features_to_df
from tsdata.index import index_data

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_comparison_data(ts_code: str = "000001.SH", lookback_days: int = 400):
    """
    获取近1年的预测对比数据
    
    Args:
        ts_code: 指数代码
        lookback_days: 回溯天数（需要比1年多，用于特征计算）
    
    Returns:
        DataFrame: 包含日期、真实值、预测值
    """
    # 加载模型
    model_dir = Path(__file__).parent / "models"
    with open(model_dir / "volatility_model_lgb.pkl", 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    print(f"模型加载成功: {len(feature_cols)} 个特征")
    
    # 获取数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    print(f"获取数据: {start_date.strftime('%Y%m%d')} 至 {end_date.strftime('%Y%m%d')}")
    
    df = index_data.get_index_daily(
        ts_code=ts_code,
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d")
    )
    
    if df is None or df.empty:
        raise ValueError(f"未获取到 {ts_code} 的数据")
    
    print(f"获取到 {len(df)} 条记录")
    
    # 特征工程
    fe = VolatilityFeatureEngineering()
    df_feat = fe.create_features(df, include_garch_features=False)
    
    # 添加GARCH特征
    print("添加GARCH特征...")
    df_feat = add_garch_features_to_df(df_feat, min_train_size=100)
    
    # 准备可用特征
    available_features = [col for col in feature_cols if col in df_feat.columns]
    print(f"可用特征: {len(available_features)}/{len(feature_cols)}")
    
    # 对每个日期进行预测
    results = []
    df_valid = df_feat.dropna(subset=available_features + ['target_vol']).copy()
    
    print(f"\n有效数据点: {len(df_valid)}")
    print("开始滚动预测...")
    
    # 确保索引是日期时间类型
    if not isinstance(df_valid.index, pd.DatetimeIndex):
        if 'trade_date' in df_valid.columns:
            df_valid['trade_date'] = pd.to_datetime(df_valid['trade_date'])
            df_valid = df_valid.set_index('trade_date')
        else:
            # 尝试将索引转换为日期时间
            df_valid.index = pd.to_datetime(df_valid.index, errors='coerce')
    
    # 只取近1年的数据（约252个交易日）
    one_year_ago = datetime.now() - timedelta(days=365)
    df_valid = df_valid[df_valid.index >= one_year_ago].copy()
    
    print(f"近1年数据点: {len(df_valid)}")
    
    for i, (date, row) in enumerate(df_valid.iterrows()):
        # 特征
        X = df_valid.loc[[date], available_features]
        
        # 预测
        pred_log = model.predict(X)[0]
        pred_vol = np.exp(pred_log)
        
        # 真实值
        true_vol = row['target_vol']
        
        results.append({
            'date': date,
            'true_vol': true_vol,
            'pred_vol': pred_vol
        })
        
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(df_valid)}")
    
    return pd.DataFrame(results)


def plot_comparison(df: pd.DataFrame, ts_code: str = "000001.SH", save_path: str = None):
    """
    绘制对比图
    """
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'])
    
    # ===== 图1: 折线对比 =====
    ax1 = axes[0]
    
    # 使用seaborn绑制
    df_melt = df.melt(id_vars=['date'], value_vars=['true_vol', 'pred_vol'],
                      var_name='type', value_name='volatility')
    df_melt['type'] = df_melt['type'].map({'true_vol': 'Actual', 'pred_vol': 'Predicted'})
    
    sns.lineplot(data=df_melt, x='date', y='volatility', hue='type', 
                 ax=ax1, linewidth=1.2, palette={'Actual': '#2E86AB', 'Predicted': '#E94F37'})
    
    ax1.set_title(f'{ts_code} Volatility Prediction Comparison (Last 1 Year)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Yang-Zhang Volatility', fontsize=11)
    ax1.legend(title='', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    ax1.tick_params(axis='x', rotation=30)
    
    # ===== 图2: 散点图（预测vs真实） =====
    ax2 = axes[1]
    
    sns.scatterplot(data=df, x='true_vol', y='pred_vol', ax=ax2, 
                    alpha=0.6, s=30, color='#E94F37', edgecolor='white', linewidth=0.5)
    
    # 添加对角线（完美预测线）
    min_vol = df[['true_vol', 'pred_vol']].min().min()
    max_vol = df[['true_vol', 'pred_vol']].max().max()
    ax2.plot([min_vol, max_vol], [min_vol, max_vol], 'k--', linewidth=1.5, label='Perfect Prediction')
    
    # 计算指标
    mae = np.mean(np.abs(df['true_vol'] - df['pred_vol']))
    rmse = np.sqrt(np.mean((df['true_vol'] - df['pred_vol']) ** 2))
    corr = df['true_vol'].corr(df['pred_vol'])
    
    # 添加指标文本
    metrics_text = f'MAE: {mae:.6f}\nRMSE: {rmse:.6f}\nCorrelation: {corr:.4f}'
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_title('Predicted vs Actual', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Actual Volatility', fontsize=11)
    ax2.set_ylabel('Predicted Volatility', fontsize=11)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存至: {save_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    # 获取数据
    df = get_comparison_data(ts_code="000001.SH", lookback_days=500)
    
    # 绑制图表
    plot_comparison(df, ts_code="000001.SH", save_path="prediction_comparison.png")
    
    # 打印统计信息
    print("\n===== 统计信息 =====")
    print(f"数据点数: {len(df)}")
    print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"\n真实值统计:")
    print(df['true_vol'].describe())
    print(f"\n预测值统计:")
    print(df['pred_vol'].describe())
    
    # 误差分析
    df['error'] = df['true_vol'] - df['pred_vol']
    df['abs_error'] = np.abs(df['error'])
    df['pct_error'] = df['abs_error'] / df['true_vol'] * 100
    
    print(f"\n误差分析:")
    print(f"MAE: {df['abs_error'].mean():.6f}")
    print(f"RMSE: {np.sqrt((df['error']**2).mean()):.6f}")
    print(f"MAPE: {df['pct_error'].mean():.2f}%")
