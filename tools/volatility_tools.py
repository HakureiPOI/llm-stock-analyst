"""
指数波动率预测工具 - LangChain工具
预测目标: 下一交易日 Yang-Zhang 波动率
"""
from langchain.tools import tool
import pandas as pd


# 全局预测器实例
_predictor_instance = None


def _get_predictor():
    """获取预测器实例"""
    global _predictor_instance
    if _predictor_instance is None:
        from ml.predict import VolatilityPredictor
        _predictor_instance = VolatilityPredictor()
    return _predictor_instance


# 常用指数
COMMON_INDICES = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000300.SH": "沪深300",
    "000016.SH": "上证50",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
}


@tool
def predict_index_volatility(ts_code: str, days: int = 300) -> str:
    """预测指数下一交易日的波动率

    基于Yang-Zhang波动率代理，预测下一交易日市场波动程度。
    波动率越高表示市场风险越大。

    Args:
        ts_code: 指数代码 (如: 000001.SH 上证指数, 000300.SH 沪深300)
        days: 获取历史数据的天数，默认300天

    Returns:
        str: 波动率预测结果，包含预测值、历史分位数和风险解读
    """
    index_name = COMMON_INDICES.get(ts_code, ts_code)

    try:
        predictor = _get_predictor()
        result = predictor.predict(ts_code, days=days, include_garch=True)
    except FileNotFoundError:
        return f"错误: 模型文件不存在，请先训练模型"
    except Exception as e:
        return f"预测失败: {str(e)}"

    # 格式化输出
    lines = []
    lines.append(f"## {index_name} ({ts_code}) 波动率预测")
    lines.append("")
    lines.append("### 下一交易日波动率预测")
    lines.append("")

    vol = result['predicted_volatility']
    percentile = result['percentile']

    lines.append(f"- **预测波动率**: {vol:.6f}")
    lines.append(f"- **历史分位数**: {percentile['value']}% ({percentile['label']})")
    lines.append(f"- **风险解读**: {percentile['description']}")
    lines.append("")

    # 历史统计
    stats = result['historical_stats']
    lines.append("### 历史波动率参考")
    lines.append(f"- 范围: [{stats['min']:.6f}, {stats['max']:.6f}]")
    lines.append(f"- 均值: {stats['mean']:.6f}")
    lines.append(f"- 中位数: {stats['q50']:.6f}")
    lines.append(f"- 25%-75%分位: [{stats['q25']:.6f}, {stats['q75']:.6f}]")
    lines.append("")

    # 模型性能
    if result.get('metrics'):
        metrics = result['metrics']
        lines.append("### 模型性能 (历史回测)")
        lines.append(f"- MAE: {metrics['mae']:.6f}")
        lines.append(f"- RMSE: {metrics['rmse']:.6f}")
        lines.append(f"- R²: {metrics['r2']:.4f}")
        lines.append(f"- 方向准确率: {metrics['direction_accuracy']:.1f}%")

    return "\n".join(lines)


@tool
def compare_index_volatility(ts_codes: str, days: int = 300) -> str:
    """对比多个指数的波动率

    对比各指数下一交易日波动率预测值及其历史分位数。

    Args:
        ts_codes: 指数代码列表，用逗号分隔 (如: "000001.SH,399006.SZ,000300.SH")
        days: 获取历史数据的天数，默认300

    Returns:
        str: 多指数波动率对比结果
    """
    try:
        predictor = _get_predictor()
    except FileNotFoundError:
        return "错误: 模型文件不存在，请先训练模型"

    codes = [c.strip() for c in ts_codes.split(',') if c.strip()]
    results = []

    for code in codes:
        try:
            result = predictor.predict(code, days=days, include_garch=True)
            results.append(result)
        except Exception:
            continue

    if not results:
        return "没有成功预测任何指数"

    # 对比分析
    lines = []
    lines.append("## 多指数波动率对比分析")
    lines.append("")
    lines.append("| 指数 | 代码 | 预测波动率 | 历史分位 | 风险等级 |")
    lines.append("|------|------|-----------|---------|---------|")

    sorted_results = sorted(results, key=lambda x: x['predicted_volatility'], reverse=True)

    for r in sorted_results:
        code = r['ts_code']
        name = COMMON_INDICES.get(code, code)
        vol = r['predicted_volatility']
        p = r['percentile']

        lines.append(f"| {name} | {code} | {vol:.6f} | {p['value']}% ({p['label']}) | {p['label']} |")

    lines.append("")

    # 综合评价
    highest = sorted_results[0]
    lowest = sorted_results[-1]

    lines.append("### 综合评价")
    lines.append(f"- **最高波动**: {COMMON_INDICES.get(highest['ts_code'], highest['ts_code'])} "
                 f"(波动率={highest['predicted_volatility']:.6f}, {highest['percentile']['label']})")
    lines.append(f"- **最低波动**: {COMMON_INDICES.get(lowest['ts_code'], lowest['ts_code'])} "
                 f"(波动率={lowest['predicted_volatility']:.6f}, {lowest['percentile']['label']})")

    return "\n".join(lines)


@tool
def get_market_volatility_summary() -> str:
    """获取市场整体波动率摘要 (主要指数快速查询)

    展示主要指数下一交易日波动率预测及历史分位数。

    Returns:
        str: 市场波动率摘要
    """
    try:
        predictor = _get_predictor()
    except FileNotFoundError:
        return "错误: 模型文件不存在，请先训练模型"

    # 主要指数
    main_indices = ["000001.SH", "399006.SZ", "000300.SH"]

    results = []
    for code in main_indices:
        try:
            result = predictor.predict(code, days=300, include_garch=True)
            results.append(result)
        except Exception:
            continue

    if not results:
        return "无法获取市场波动率数据"

    # 格式化输出
    lines = []
    lines.append("## 市场波动率摘要")
    lines.append("")

    for r in results:
        code = r['ts_code']
        name = COMMON_INDICES.get(code, code)
        vol = r['predicted_volatility']
        p = r['percentile']

        lines.append(f"### {name}")
        lines.append(f"- 预测波动率: **{vol:.6f}**")
        lines.append(f"- 历史分位: {p['value']}% ({p['label']})")
        lines.append(f"- 风险解读: {p['description']}")
        lines.append("")

    # 整体判断
    avg_vol = sum(r['predicted_volatility'] for r in results) / len(results)
    avg_pct = sum(r['percentile']['value'] for r in results) / len(results)

    if avg_pct < 25:
        overall = "整体市场波动率处于历史低位，投资环境相对稳定"
    elif avg_pct < 50:
        overall = "整体市场波动率低于历史中位，风险较低"
    elif avg_pct < 75:
        overall = "整体市场波动率处于历史中等水平"
    elif avg_pct < 90:
        overall = "整体市场波动率偏高，建议控制仓位"
    else:
        overall = "整体市场波动率处于历史高位，建议谨慎观望"

    lines.append("### 市场整体判断")
    lines.append(f"- 平均波动率: {avg_vol:.6f}")
    lines.append(f"- 平均分位数: {avg_pct:.1f}%")
    lines.append(f"- 综合评价: {overall}")

    return "\n".join(lines)
