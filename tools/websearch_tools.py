"""联网搜索工具 - 基于通义千问的联网能力"""
import os
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
                "content": "你是一个专业搜索助手。请使用联网搜索获取最新信息，并整理成清晰的格式返回。"
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
def web_search(query: str) -> str:
    """联网搜索工具，获取实时互联网信息。

    【重要】此工具仅用于以下场景：
    - 用户明确要求查询"最新新闻"、"最新公告"、"最新政策"
    - 市场热点事件、宏观政策等无法从结构化数据获取的信息
    - 用户要求的交叉验证

    【禁止】用于以下场景（请使用专家智能体）：
    - 查询股票/指数基础信息、K线数据、财务数据、技术指标
    - 查询指数成分股、风险度预测

    Args:
        query: 搜索关键词或问题

    Returns:
        搜索结果摘要
    """
    if not DASHSCOPE_API_KEY:
        return "错误: 未配置 DASHSCOPE_API_KEY 环境变量，无法使用联网搜索功能。"

    try:
        search_result = _do_web_search(query)
        return f"联网搜索结果:\n{'=' * 50}\n{search_result}"

    except ImportError:
        return "错误: 未安装 dashscope 库，请运行: pip install dashscope"
    except Exception as e:
        return f"联网搜索出错: {str(e)}"


__all__ = ["web_search"]
