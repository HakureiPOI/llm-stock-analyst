"""模型配置模块"""
import os
import dotenv
from typing import Optional
from langchain.chat_models import init_chat_model

dotenv.load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 启动时验证必要的 API Key
if not DASHSCOPE_API_KEY:
    raise ValueError(
        "DASHSCOPE_API_KEY 环境变量未设置，请在 .env 文件中配置。\n"
        "示例：DASHSCOPE_API_KEY=your_api_key_here"
    )


def get_chat_model(
    model: str = "qwen3.5-plus",
    model_provider: str = "openai",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    获取聊天模型实例

    Args:
        model: 模型名称，默认 "qwen3.5-plus"
        model_provider: 模型提供商，默认 "openai"
        temperature: 温度参数，默认 0.7
        api_key: API 密钥，默认从环境变量 DASHSCOPE_API_KEY 读取
        base_url: API 基础 URL，默认阿里云 DashScope 兼容接口

    Returns:
        聊天模型实例
        
    Raises:
        ValueError: 当 API Key 未设置时
    """
    if api_key is None:
        api_key = DASHSCOPE_API_KEY
    
    if not api_key:
        raise ValueError("API Key 未设置，请检查 DASHSCOPE_API_KEY 环境变量")

    if base_url is None:
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    return init_chat_model(
        model=model,
        model_provider=model_provider,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


# 默认模型配置
DEFAULT_MODEL_CONFIG = {
    "model": "qwen3.5-plus",
    "model_provider": "openai",
    "temperature": 0.7,
}


def get_default_chat_model():
    """获取默认配置的聊天模型"""
    return get_chat_model(**DEFAULT_MODEL_CONFIG)
