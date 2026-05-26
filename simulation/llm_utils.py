"""LLM 工具函数"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import config


def get_llm() -> ChatOpenAI:
    """获取 LLM 实例"""
    kwargs = {
        "model": config.MODEL_NAME,
        "api_key": config.OPENAI_API_KEY,
        "temperature": 0.8,
        "max_tokens": 1024,
    }
    if config.OPENAI_API_BASE:
        kwargs["base_url"] = config.OPENAI_API_BASE
    return ChatOpenAI(**kwargs)


async def llm_call(system_prompt: str, user_prompt: str) -> str:
    """异步调用 LLM"""
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = await llm.ainvoke(messages)
    return response.content


def llm_call_sync(system_prompt: str, user_prompt: str) -> str:
    """同步调用 LLM"""
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content
