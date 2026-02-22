"""
股票波动率预测工具 - LangChain工具
"""
from langchain.tools import tool
import pandas as pd


# 全局预测器实例（避免重复加载）
_predictor_instance = None


def _get_predictor():
    """获取预测器实例"""
    global _predictor_instance
    if _predictor_instance is None:
        from ml.predict import VolatilityPredictor
        _predictor_instance = VolatilityPredictor()
    return _predictor_instance


@tool
def predict_stock_volatility(ts_code: str, days: int = 365, window_size: int = 5) -> str:
    """预测单只股票的未来波动率

    Args:
        ts_code: 股票代码 (如: 600519.SH)
        days: 获取历史数据的天数，默认365天
        window_size: 波动率计算窗口，默认5天

    Returns:
        str: 波动率预测结果，包含预测数据和模型性能指标
    """
    predictor = _get_predictor()
    result = predictor.predict(ts_code, days=days, window_size=window_size)

    if 'error' in result:
        return f"预测失败: {result['error']}"

    # 格式化输出
    lines = []
    lines.append(f"## {ts_code} 波动率预测结果")
    lines.append("")
    lines.append("### 数据概况")
    lines.append(f"- 时间范围: {result['data_period']}")
    lines.append(f"- 总记录数: {result['total_records']}")
    lines.append(f"- 有效预测数: {result['valid_predictions']}")
    lines.append("")

    lines.append("### 模型性能")
    metrics = result['metrics']
    lines.append(f"- MAE:  {metrics['mae']:.6f}")
    lines.append(f"- RMSE: {metrics['rmse']:.6f}")
    lines.append(f"- R²:   {metrics['r2']:.4f}")
    lines.append("")

    # 最近预测
    if result.get('actual_vs_predicted'):
        lines.append("### 最近5天预测对比")
        lines.append("| 日期 | 实际波动率 | 预测波动率 | 误差 |")
        lines.append("|------|-----------|-----------|------|")

        for date, actual, pred in result['actual_vs_predicted'][-5:]:
            error = abs(actual - pred) if not pd.isna(actual) and not pd.isna(pred) else 0
            lines.append(f"| {date} | {actual:.6f} | {pred:.6f} | {error:.6f} |")
        lines.append("")

    return "\n".join(lines)


@tool
def compare_stock_volatility(stock_codes: str, days: int = 365, window_size: int = 5) -> str:
    """对比多只股票的波动率预测结果

    Args:
        stock_codes: 股票代码列表，用逗号分隔（如: "600519.SH,601318.SH,600036.SH"）
        days: 获取历史数据的天数，默认365
        window_size: 波动率计算窗口，默认5

    Returns:
        str: 多股票波动率对比结果
    """
    import pandas as pd

    predictor = _get_predictor()
    codes = [c.strip() for c in stock_codes.split(',') if c.strip()]

    results = []
    for code in codes:
        try:
            result = predictor.predict(code, days=days, window_size=window_size)
            if 'metrics' in result and result['metrics']['r2'] is not None:
                results.append(result)
        except Exception:
            continue

    if not results:
        return "没有成功预测任何股票"

    # 对比分析
    lines = []
    lines.append("## 多股票波动率对比分析")
    lines.append("")
    lines.append("### 各股票预测表现")
    lines.append("| 股票代码 | R² | RMSE | 平均预测波动率 |")
    lines.append("|---------|----|------|----------------|")

    for r in results:
        avg_pred = sum(p['predicted_volatility'] for p in r['predictions']) / len(r['predictions']) if r['predictions'] else 0
        lines.append(f"| {r['ts_code']} | {r['metrics']['r2']:.4f} | {r['metrics']['rmse']:.6f} | {avg_pred:.6f} |")
    lines.append("")

    # 找出最优和最差
    best = max(results, key=lambda x: x['metrics']['r2'])
    worst = min(results, key=lambda x: x['metrics']['r2'])

    lines.append("### 综合评价")
    lines.append(f"- 模型表现最佳: {best['ts_code']} (R²={best['metrics']['r2']:.4f})")
    lines.append(f"- 模型表现较差: {worst['ts_code']} (R²={worst['metrics']['r2']:.4f})")
    lines.append("")

    return "\n".join(lines)


@tool
def get_volatility_forecast_summary(ts_code: str, days: int = 365, forecast_days: int = 5) -> str:
    """获取股票波动率预测摘要（快速查询）

    Args:
        ts_code: 股票代码
        days: 获取历史数据的天数，默认365
        forecast_days: 预测未来天数，默认5

    Returns:
        str: 波动率预测摘要
    """
    import pandas as pd

    predictor = _get_predictor()
    result = predictor.predict(ts_code, days=days)

    if 'error' in result:
        return f"预测失败: {result['error']}"

    # 获取最近的预测值
    recent_predictions = result['predictions'][-forecast_days:]
    forecast_values = [p['predicted_volatility'] for p in recent_predictions]

    # 计算波动率趋势
    if len(forecast_values) >= 3:
        recent_avg = sum(forecast_values[-3:]) / 3
        earlier_avg = sum(forecast_values[-6:-3]) / len(forecast_values[-6:-3]) if len(forecast_values) >= 6 else recent_avg

        if recent_avg > earlier_avg * 1.05:
            trend = "上升"
        elif recent_avg < earlier_avg * 0.95:
            trend = "下降"
        else:
            trend = "稳定"
    else:
        trend = "未知"

    # 评估模型置信度
    r2 = result['metrics']['r2']
    if r2 > 0.7:
        confidence = "高"
    elif r2 > 0.5:
        confidence = "中"
    else:
        confidence = "低"

    # 格式化输出
    lines = []
    lines.append(f"## {ts_code} 波动率预测摘要")
    lines.append("")
    lines.append(f"- 当前预测波动率: {forecast_values[-1]:.6f}" if forecast_values else "- 当前预测波动率: N/A")
    lines.append(f"- 波动率趋势: {trend}")
    lines.append(f"- 模型置信度: {confidence} (R²={r2:.4f})")
    lines.append(f"- 预测数据点数: {result['valid_predictions']}")
    lines.append("")

    return "\n".join(lines)

