"""获取指数日线数据"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tsdata.index import index_data
from tsdata.cache import clear_cache


# 常用指数列表
COMMON_INDICES = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000300.SH": "沪深300",
    "000016.SH": "上证50",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
}


def get_index_daily_data(ts_code: str = "000001.SH", years: int = 5) -> pd.DataFrame:
    """
    获取单个指数的日线数据

    Args:
        ts_code: 指数代码
        years: 获取年数

    Returns:
        DataFrame: 指数日线数据
    """
    # 清空缓存
    clear_cache()

    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")

    print(f"获取 {ts_code} 数据: {start_date_str} 至 {end_date_str}")

    # 获取数据
    df = index_data.get_index_daily(
        ts_code=ts_code,
        start_date=start_date_str,
        end_date=end_date_str
    )

    if df is None or df.empty:
        raise ValueError(f"未获取到 {ts_code} 的数据")

    # 按日期排序
    df = df.sort_values('trade_date').reset_index(drop=True)

    print(f"获取到 {len(df)} 条记录")

    return df


def save_index_data(ts_code: str = "000001.SH", years: int = 5) -> str:
    """
    获取并保存指数数据

    Args:
        ts_code: 指数代码
        years: 获取年数

    Returns:
        str: 保存文件路径
    """
    df = get_index_daily_data(ts_code, years)

    # 保存
    ml_dir = Path(__file__).parent
    dataset_dir = ml_dir / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    save_path = dataset_dir / f"index_daily_{ts_code.replace('.', '_')}_{years}years.csv"
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"已保存到: {save_path}")

    return str(save_path)


def get_multi_index_data(ts_codes: list, years: int = 5) -> pd.DataFrame:
    """
    获取多个指数的日线数据并合并

    Args:
        ts_codes: 指数代码列表
        years: 获取年数

    Returns:
        DataFrame: 合并后的指数数据
    """
    all_data = []

    for ts_code in ts_codes:
        try:
            df = get_index_daily_data(ts_code, years)
            all_data.append(df)
        except Exception as e:
            print(f"获取 {ts_code} 失败: {e}")

    if not all_data:
        raise ValueError("未获取到任何数据")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    print(f"合并完成: {len(combined)} 条记录, {combined['ts_code'].nunique()} 个指数")

    return combined


if __name__ == "__main__":
    # 示例: 获取上证指数5年数据
    save_path = save_index_data(ts_code="000001.SH", years=5)
