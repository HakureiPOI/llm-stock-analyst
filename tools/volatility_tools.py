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

    返回原始预测数据，包括：
    - 预测波动率值
    - 多维度分位数统计（滚动窗口）
    - 语义指标（偏离度、趋势、聚类状态、风险等级）
    - SHAP特征贡献度
    - 模型性能指标

    Args:
        ts_code: 指数代码 (如: 000001.SH, 000300.SH)
        days: 获取历史数据的天数，默认300天

    Returns:
        str: JSON格式的波动率预测数据
    """
    index_name = COMMON_INDICES.get(ts_code, ts_code)

    try:
        predictor = _get_predictor()
        result = predictor.predict(ts_code, days=days, include_garch=True, include_shap=True)
    except FileNotFoundError:
        return f"错误: 模型文件不存在，请先训练模型"
    except Exception as e:
        return f"预测失败: {str(e)}"

    # 返回结构化数据
    import json
    return json.dumps({
        "index_name": index_name,
        "ts_code": ts_code,
        "predicted_volatility": result['predicted_volatility'],
        "volatility_pct": result['volatility_pct'],
        "percentile": result['percentile'],
        "historical_stats": result['historical_stats'],
        "volatility_trend": result.get('volatility_trend'),
        "semantic_metrics": result.get('semantic_metrics'),
        "shap_analysis": result.get('shap_analysis'),
        "model_metrics": result.get('model_metrics'),
    }, ensure_ascii=False, indent=2)


@tool
def compare_index_volatility(ts_codes: str, days: int = 300) -> str:
    """对比多个指数的波动率预测数据

    Args:
        ts_codes: 指数代码列表，逗号分隔 (如: "000001.SH,399006.SZ,000300.SH")
        days: 获取历史数据的天数，默认300

    Returns:
        str: JSON格式的多指数波动率对比数据
    """
    try:
        predictor = _get_predictor()
    except FileNotFoundError:
        return "错误: 模型文件不存在，请先训练模型"

    codes = [c.strip() for c in ts_codes.split(',') if c.strip()]
    results = []

    for code in codes:
        try:
            result = predictor.predict(code, days=days, include_garch=True, include_shap=False)
            results.append({
                "ts_code": code,
                "index_name": COMMON_INDICES.get(code, code),
                "predicted_volatility": result['predicted_volatility'],
                "volatility_pct": result['volatility_pct'],
                "percentile": result['percentile'],
                "volatility_trend": result.get('volatility_trend'),
                "semantic_metrics": result.get('semantic_metrics'),
            })
        except Exception:
            continue

    if not results:
        return "错误: 没有成功预测任何指数"

    import json
    return json.dumps({
        "indices": results,
        "count": len(results),
    }, ensure_ascii=False, indent=2)


@tool
def get_market_volatility_summary() -> str:
    """获取市场主要指数波动率预测数据

    Returns:
        str: JSON格式的市场波动率数据汇总
    """
    try:
        predictor = _get_predictor()
    except FileNotFoundError:
        return "错误: 模型文件不存在，请先训练模型"

    main_indices = ["000001.SH", "399006.SZ", "000300.SH"]
    results = []

    for code in main_indices:
        try:
            result = predictor.predict(code, days=300, include_garch=True, include_shap=True)
            results.append({
                "ts_code": code,
                "index_name": COMMON_INDICES.get(code, code),
                "predicted_volatility": result['predicted_volatility'],
                "volatility_pct": result['volatility_pct'],
                "percentile": result['percentile'],
                "volatility_trend": result.get('volatility_trend'),
                "semantic_metrics": result.get('semantic_metrics'),
                "shap_analysis": result.get('shap_analysis'),
            })
        except Exception:
            continue

    if not results:
        return "错误: 无法获取市场波动率数据"

    # 计算统计数据
    avg_vol = sum(r['predicted_volatility'] for r in results) / len(results)
    avg_pct = sum(r['percentile'] for r in results) / len(results)

    import json
    return json.dumps({
        "indices": results,
        "summary": {
            "avg_volatility": avg_vol,
            "avg_percentile": avg_pct,
            "index_count": len(results),
        },
    }, ensure_ascii=False, indent=2)
