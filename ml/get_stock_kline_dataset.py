import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tsdata.stock import stock_data
from tsdata.cache import clear_cache


def get_stock_kline_dataset(pool_file=None, index_code="000016.SH", years=5):
    """获取股票池中每只股票近N年的日线数据并合并保存"""
    # 清空缓存避免bug
    clear_cache()

    # 确定股票池文件路径
    if pool_file is None:
        ml_dir = Path(__file__).parent
        pool_file = ml_dir / "dataset" / f"stock_pool_{index_code.replace('.', '_')}.csv"

    # 读取股票池
    stock_pool = pd.read_csv(pool_file)

    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")

    # 获取每只股票的数据
    all_data = []
    for _, row in stock_pool.iterrows():
        ts_code = row['con_code']
        df = stock_data.get_daily(
            ts_code=ts_code,
            start_date=start_date_str,
            end_date=end_date_str
        )

        if df is not None and not df.empty:
            # 修复: 确保ts_code正确
            df['ts_code'] = ts_code
            all_data.append(df)

    # 合并并去重
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.drop_duplicates()
    combined_df = combined_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 保存
    ml_dir = Path(__file__).parent
    dataset_dir = ml_dir / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    save_path = dataset_dir / f"kline_dataset_{index_code.replace('.', '_')}_{years}years.csv"
    combined_df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"完成: {len(combined_df)} 条记录, {combined_df['ts_code'].nunique()} 只股票")
    print(f"已保存到: {save_path}")

    return combined_df


if __name__ == "__main__":
    get_stock_kline_dataset(index_code="000016.SH", years=5)
