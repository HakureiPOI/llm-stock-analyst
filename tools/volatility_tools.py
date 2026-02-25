"""
指数风险度预测工具 - LangChain工具
"""
from langchain.tools import tool
import pandas as pd


# 全局预测器实例
_predictor_instance = None


def _get_predictor():
    """获取预测器实例"""
    global _predictor_instance
    if _predictor_instance is None:
        from ml.predict import IndexRiskPredictor
        _predictor_instance = IndexRiskPredictor()
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
def predict_index_risk(ts_code: str, days: int = 500) -> str:
    """预测指数未来风险度 (已实现波动率)

    风险度越高表示指数波动越剧烈，投资风险越大。
    同时预测 5日(一周) 和 20日(一月) 两个时间窗口的风险度，
    并提供历史分位数语义解读。

    Args:
        ts_code: 指数代码 (如: 000001.SH 上证指数, 000300.SH 沪深300)
        days: 获取历史数据的天数，默认500天

    Returns:
        str: 风险度预测结果，包含预测值、历史分位数和语义解读
    """
    # 获取指数名称
    index_name = COMMON_INDICES.get(ts_code, ts_code)

    try:
        predictor = _get_predictor()
        result = predictor.predict(ts_code, days=days, horizons=[5, 20])
    except FileNotFoundError:
        return f"错误: 模型文件不存在，请先训练模型"
    except Exception as e:
        return f"预测失败: {str(e)}"

    # 格式化输出
    lines = []
    lines.append(f"## {index_name} ({ts_code}) 风险度预测")
    lines.append("")

    # 最新预测 - 两个时间窗口
    lines.append("### 未来风险度预测")
    lines.append("")
    
    for horizon in [5, 20]:
        if horizon in result['latest_predictions']:
            rv = result['latest_predictions'][horizon]
            period = "一周" if horizon == 5 else "一月"
            
            # 分位数语义
            percentile_info = ""
            if horizon in result.get('percentiles', {}):
                p = result['percentiles'][horizon]
                percentile_info = f" (历史分位: {p['percentile']}% - {p['label']})"
            
            # 风险等级判断
            if rv < 0.015:
                risk_level = "低风险"
                risk_desc = "市场波动较小，适合稳健投资"
            elif rv < 0.025:
                risk_level = "中等风险"
                risk_desc = "市场波动正常，可保持原有策略"
            elif rv < 0.04:
                risk_level = "较高风险"
                risk_desc = "市场波动较大，建议适当控制仓位"
            else:
                risk_level = "高风险"
                risk_desc = "市场波动剧烈，建议谨慎操作"

            lines.append(f"**未来{horizon}日 ({period})**{percentile_info}:")
            lines.append(f"- 已实现波动率: **{rv:.6f}**")
            lines.append(f"- 风险等级: {risk_level}")
            
            # 分位数描述
            if horizon in result.get('percentiles', {}):
                lines.append(f"- 语义解读: {result['percentiles'][horizon]['description']}")
            else:
                lines.append(f"- 风险描述: {risk_desc}")
            
            # 历史统计参考
            if horizon in result.get('historical_stats', {}):
                stats = result['historical_stats'][horizon]
                lines.append(f"- 历史参考: 中位数 {stats['q50']:.6f}, 范围 [{stats['min']:.6f}, {stats['max']:.6f}]")
            
            lines.append("")

    # 数据概况
    lines.append("### 数据概况")
    lines.append(f"- 时间范围: {result['data_period']}")
    lines.append(f"- 有效预测数: {result['valid_predictions']}")
    lines.append("")

    # 模型性能
    lines.append("### 模型性能")
    for horizon in [5, 20]:
        if horizon in result['metrics']:
            metrics = result['metrics'][horizon]
            period = "一周" if horizon == 5 else "一月"
            if metrics['mae'] is not None:
                lines.append(f"- {horizon}日 ({period}): MAE={metrics['mae']:.6f}, R²={metrics['r2']:.4f}")
            else:
                lines.append(f"- {horizon}日 ({period}): (无评估数据)")

    return "\n".join(lines)


@tool
def compare_index_risk(ts_codes: str, days: int = 500) -> str:
    """对比多个指数的风险度

    同时对比 5日(一周) 和 20日(一月) 两个时间窗口的风险度，
    并展示各指数的历史分位数。

    Args:
        ts_codes: 指数代码列表，用逗号分隔 (如: "000001.SH,399006.SZ,000300.SH")
        days: 获取历史数据的天数，默认500

    Returns:
        str: 多指数风险度对比结果
    """
    try:
        predictor = _get_predictor()
    except FileNotFoundError:
        return "错误: 模型文件不存在，请先训练模型"

    codes = [c.strip() for c in ts_codes.split(',') if c.strip()]
    results = []

    for code in codes:
        try:
            result = predictor.predict(code, days=days, horizons=[5, 20])
            if 'latest_predictions' in result:
                results.append(result)
        except Exception:
            continue

    if not results:
        return "没有成功预测任何指数"

    # 对比分析
    lines = []
    lines.append("## 多指数风险度对比分析")
    lines.append("")
    
    for horizon in [5, 20]:
        period = "一周" if horizon == 5 else "一月"
        lines.append(f"### {horizon}日 ({period}) 风险度预测")
        lines.append("| 指数 | 代码 | 预测波动率 | 历史分位 | 风险等级 |")
        lines.append("|------|------|-----------|---------|---------|")

        sorted_results = sorted(results, key=lambda x: x['latest_predictions'].get(horizon, 0), reverse=True)
        
        for r in sorted_results:
            code = r['ts_code']
            name = COMMON_INDICES.get(code, code)
            rv = r['latest_predictions'].get(horizon, 0)
            
            # 分位数
            percentile_str = "-"
            if horizon in r.get('percentiles', {}):
                p = r['percentiles'][horizon]
                percentile_str = f"{p['percentile']}% ({p['label']})"

            if rv < 0.015:
                level = "低"
            elif rv < 0.025:
                level = "中"
            elif rv < 0.04:
                level = "较高"
            else:
                level = "高"

            lines.append(f"| {name} | {code} | {rv:.6f} | {percentile_str} | {level} |")
        lines.append("")

    # 综合评价 - 基于5日预测
    if results:
        highest = max(results, key=lambda x: x['latest_predictions'].get(5, 0))
        lowest = min(results, key=lambda x: x['latest_predictions'].get(5, 0))

        lines.append("### 综合评价 (基于一周风险度)")
        lines.append(f"- **最高风险**: {COMMON_INDICES.get(highest['ts_code'], highest['ts_code'])} "
                     f"(波动率={highest['latest_predictions'].get(5, 0):.6f})")
        lines.append(f"- **最低风险**: {COMMON_INDICES.get(lowest['ts_code'], lowest['ts_code'])} "
                     f"(波动率={lowest['latest_predictions'].get(5, 0):.6f})")
        lines.append("")

    return "\n".join(lines)


@tool
def get_market_risk_summary() -> str:
    """获取市场整体风险摘要 (主要指数风险度快速查询)

    同时展示 5日(一周) 和 20日(一月) 两个时间窗口的风险度，
    以及历史分位数语义。

    Returns:
        str: 市场风险摘要
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
            result = predictor.predict(code, days=500, horizons=[5, 20])
            if 'latest_predictions' in result:
                results.append(result)
        except Exception:
            continue

    if not results:
        return "无法获取市场风险数据"

    # 格式化输出
    lines = []
    lines.append("## 市场风险摘要")
    lines.append("")

    for r in results:
        code = r['ts_code']
        name = COMMON_INDICES.get(code, code)

        lines.append(f"### {name}")
        
        for horizon in [5, 20]:
            rv = r['latest_predictions'].get(horizon, 0)
            period = "一周" if horizon == 5 else "一月"

            # 分位数信息
            percentile_str = ""
            if horizon in r.get('percentiles', {}):
                p = r['percentiles'][horizon]
                percentile_str = f" | 分位: {p['percentile']}% ({p['label']})"

            if rv < 0.015:
                level = "🟢 低风险"
            elif rv < 0.025:
                level = "🟡 中等风险"
            elif rv < 0.04:
                level = "🟠 较高风险"
            else:
                level = "🔴 高风险"

            lines.append(f"- {period}风险度: **{rv:.6f}** ({level}){percentile_str}")
        
        lines.append("")

    # 整体风险判断 - 基于5日预测
    avg_rv = sum(r['latest_predictions'].get(5, 0) for r in results) / len(results)
    if avg_rv < 0.015:
        overall = "整体市场波动较小，投资环境相对稳定"
    elif avg_rv < 0.025:
        overall = "整体市场波动正常，可按原策略操作"
    elif avg_rv < 0.04:
        overall = "整体市场波动较大，建议控制仓位"
    else:
        overall = "整体市场波动剧烈，建议谨慎观望"

    lines.append("### 市场整体判断")
    lines.append(overall)

    return "\n".join(lines)
