import os
import pandas as pd
from pathlib import Path
from tsdata.index import index_data


def get_stock_pool(index_code="000016.SH"):
    # 获取指数成分股权重数据
    df = index_data.get_index_weight(index_code=index_code)
    
    if df is None or df.empty:
        raise ValueError(f"未获取到指数 {index_code} 的成分股数据")
    
    # 获取最新交易日期的数据
    latest_date = df['trade_date'].max()
    df_latest = df[df['trade_date'] == latest_date].copy()
    
    # 按权重降序排列
    df_latest = df_latest.sort_values('weight', ascending=False).reset_index(drop=True)
    
    # 保存到本地
    ml_dir = Path(__file__).parent
    dataset_dir = ml_dir / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    save_path = dataset_dir / f"stock_pool_{index_code.replace('.', '_')}.csv"
    df_latest.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print(f"\n股票池已保存到: {save_path}")
    
    return df_latest


if __name__ == "__main__":
    # 获取上证50股票池
    pool_df = get_stock_pool(index_code="000016.SH")