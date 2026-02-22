"""联网搜索工具 - 基于通义千问的联网能力"""
import os
import json
from langchain.tools import tool

# 从环境变量获取 API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")


def _do_web_search(search_query: str) -> str:
    """执行实际的联网搜索"""
    import dashscope
    from dashscope import Generation

    dashscope.api_key = DASHSCOPE_API_KEY

    # 使用通义千问的联网搜索功能（模型会自动搜索）
    response = Generation.call(
        model="qwen-max",
        messages=[
            {
                "role": "system",
                "content": "你是一个专业搜索助手。当用户询问任何问题时，请使用联网搜索获取最新信息。"
            },
            {
                "role": "user",
                "content": search_query
            }
        ],
        result_format="message",
        enable_search=True  # 启用联网搜索
    )

    if response.status_code == 200:
        content = response.output.choices[0].message.content
        return content
    else:
        return f"搜索失败: {response.message}"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """使用通义千问的联网搜索能力，获取实时互联网信息。

    适用于以下场景:
    - 获取最新的新闻资讯
    - 查询实时市场动态
    - 搜索公司最新公告
    - 获取宏观经济数据
    - 查询行业研究报告
    - 补充实时信息到分析中

    Args:
        query: 搜索关键词或问题，越具体效果越好
        max_results: 返回结果数量，默认5条，范围1-10

    Returns:
        搜索结果摘要，包含标题、来源URL和简短描述
    """
    if not DASHSCOPE_API_KEY:
        return "错误: 未配置 DASHSCOPE_API_KEY 环境变量，无法使用联网搜索功能。"

    if max_results < 1 or max_results > 10:
        max_results = min(max(1, max_results), 10)

    try:
        # 执行搜索
        search_result = _do_web_search(query)

        return f"联网搜索结果:\n{'=' * 60}\n搜索关键词: {query}\n{'=' * 60}\n\n{search_result}"

    except ImportError:
        return "错误: 未安装 dashscope 库，请运行: pip install dashscope"
    except Exception as e:
        return f"联网搜索出错: {str(e)}"


@tool
def web_search_company(ts_code: str, query_type: str = "news") -> str:
    """搜索特定公司的相关信息。

    Args:
        ts_code: 股票代码 (如 600519.SH)
        query_type: 搜索类型，可选值:
                   - "news": 最新新闻
                   - "announcement": 公告信息
                   - "analysis": 分析报告
                   - "general": 综合信息

    Returns:
        搜索到的公司相关信息
    """
    type_keywords = {
        "news": "最新新闻",
        "announcement": "公告",
        "analysis": "分析报告",
        "general": "信息"
    }

    keyword = type_keywords.get(query_type, "信息")

    # 从 ts_code 提取股票代码数字部分作为搜索关键词
    code = ts_code.split('.')[0]

    search_query = f"{keyword} {code}"
    return web_search.invoke({'query': search_query, 'max_results': 5})


@tool
def web_search_market(index_code: str = "000001.SH") -> str:
    """搜索大盘指数的最新市场动态和资讯。

    Args:
        index_code: 指数代码，默认上证指数 (000001.SH)

    Returns:
        市场动态搜索结果
    """
    index_names = {
        "000001.SH": "上证指数",
        "399001.SZ": "深证成指",
        "399006.SZ": "创业板指",
        "000300.SH": "沪深300",
        "000016.SH": "上证50",
        "000905.SH": "中证500"
    }

    index_name = index_names.get(index_code, "大盘指数")
    search_query = f"{index_name} 最新行情 分析"
    return web_search.invoke({'query': search_query, 'max_results': 5})


__all__ = [
    "web_search",
    "web_search_company",
    "web_search_market",
]
