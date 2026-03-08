"""股票推荐工具 - 基于机构观点"""
import os
import json
from langchain.tools import tool
from typing import Optional

# 从环境变量获取 API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")


def _do_web_search(search_query: str) -> str:
    """执行实际的联网搜索"""
    import dashscope
    from dashscope import Generation

    dashscope.api_key = DASHSCOPE_API_KEY

    response = Generation.call(
        model="qwen-max",
        messages=[
            {
                "role": "system",
                "content": "你是一个专业搜索助手。请使用联网搜索获取最新信息，并整理成清晰的格式返回。"
            },
            {
                "role": "user",
                "content": search_query
            }
        ],
        result_format="message",
        enable_search=True
    )

    if response.status_code == 200:
        content = response.output.choices[0].message.content
        return content
    else:
        return f"搜索失败: {response.message}"


@tool
def search_institution_recommendations(sector: Optional[str] = None) -> str:
    """搜索近期机构推荐的A股个股。

    通过联网搜索获取券商、基金等机构的最新推荐观点，
    包括推荐股票代码、推荐机构、推荐理由等信息。

    Args:
        sector: 可选的行业/板块筛选，如"科技"、"消费"、"医药"等。
                不传则搜索综合推荐。

    Returns:
        机构推荐股票列表的JSON字符串，包含：
        - 推荐股票代码和名称
        - 推荐机构
        - 推荐理由
        - 来源时间
    """
    if not DASHSCOPE_API_KEY:
        return json.dumps({
            "error": "未配置 DASHSCOPE_API_KEY 环境变量",
            "recommendations": []
        }, ensure_ascii=False)

    # 构建搜索关键词
    if sector:
        query = f"近期机构推荐{sector}板块A股 券商金股 投资建议"
    else:
        query = "近期机构推荐A股 券商金股 投资建议 最新"

    try:
        # 执行搜索
        search_result = _do_web_search(query)

        # 让LLM提取结构化信息
        extract_query = f"""
请从以下搜索结果中提取机构推荐的A股信息，以JSON格式返回：

{search_result}

要求返回格式：
{{
    "recommendations": [
        {{
            "stock_code": "股票代码，如600519.SH",
            "stock_name": "股票名称",
            "institutions": ["推荐机构列表"],
            "reasons": ["推荐理由"],
            "target_price": "目标价（如有）",
            "source_date": "信息来源日期"
        }}
    ],
    "summary": "整体推荐趋势总结",
    "search_time": "搜索时间"
}}

注意：
1. 只提取明确提到的A股股票
2. 股票代码格式统一为6位数字+交易所后缀(如.SH或.SZ)
3. 如果没有明确信息，返回空列表
4. 只返回JSON，不要其他解释
"""
        extracted = _do_web_search(extract_query)

        # 尝试解析JSON
        try:
            # 提取JSON部分
            json_start = extracted.find('{')
            json_end = extracted.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = extracted[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = {
                    "recommendations": [],
                    "raw_content": extracted
                }
        except json.JSONDecodeError:
            result = {
                "recommendations": [],
                "raw_content": extracted
            }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except ImportError:
        return json.dumps({
            "error": "未安装 dashscope 库",
            "recommendations": []
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": f"搜索出错: {str(e)}",
            "recommendations": []
        }, ensure_ascii=False)


@tool
def search_hot_stocks(market: str = "A股") -> str:
    """搜索当前市场热门股票。

    通过联网搜索获取当前市场关注度高的股票，
    包括热门概念股、资金流入股等。

    Args:
        market: 市场类型，默认"A股"，可选"港股"、"美股"

    Returns:
        热门股票列表的JSON字符串
    """
    if not DASHSCOPE_API_KEY:
        return json.dumps({
            "error": "未配置 DASHSCOPE_API_KEY 环境变量",
            "hot_stocks": []
        }, ensure_ascii=False)

    query = f"{market}热门股票 资金流入 概念股 今日关注"

    try:
        search_result = _do_web_search(query)

        extract_query = f"""
请从以下搜索结果中提取{market}热门股票信息，以JSON格式返回：

{search_result}

要求返回格式：
{{
    "hot_stocks": [
        {{
            "stock_code": "股票代码",
            "stock_name": "股票名称",
            "hot_reason": "热门原因",
            "change_pct": "涨跌幅（如有）",
            "volume_ratio": "量比（如有）",
            "concept": "所属概念"
        }}
    ],
    "market_sentiment": "市场情绪总结"
}}

只返回JSON，不要其他解释。
"""
        extracted = _do_web_search(extract_query)

        try:
            json_start = extracted.find('{')
            json_end = extracted.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = extracted[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = {"hot_stocks": [], "raw_content": extracted}
        except json.JSONDecodeError:
            result = {"hot_stocks": [], "raw_content": extracted}

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"搜索出错: {str(e)}",
            "hot_stocks": []
        }, ensure_ascii=False)


__all__ = ["search_institution_recommendations", "search_hot_stocks"]
