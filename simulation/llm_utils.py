"""LLM 工具函数 - 带速率限制和重试"""

import time
import logging
import threading
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import config

logger = logging.getLogger(__name__)

# ─── 全局速率控制 ─────────────────────────────────────────────
# 同时并发的 LLM 请求数限制（OpenAI 兼容 API 通常允许少量并发）
_MAX_CONCURRENT = 3
_semaphore = threading.Semaphore(_MAX_CONCURRENT)

# 请求间最小间隔（秒），防止瞬间打爆速率
_MIN_INTERVAL = 4.0
_last_request_time = 0.0
_interval_lock = threading.Lock()


def _wait_for_rate_limit():
    """确保两次请求之间至少间隔 _MIN_INTERVAL 秒"""
    global _last_request_time
    with _interval_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < _MIN_INTERVAL:
            wait = _MIN_INTERVAL - elapsed
            logger.debug("Rate limiter: waiting %.2fs", wait)
            time.sleep(wait)
        _last_request_time = time.time()


def get_llm() -> ChatOpenAI:
    """获取 LLM 实例"""
    kwargs = {
        "model": config.MODEL_NAME,
        "api_key": config.OPENAI_API_KEY,
        "temperature": 0.8,
        "max_tokens": 1024,
        "max_retries": 0,  # 我们自己管理重试
    }
    if config.OPENAI_API_BASE:
        kwargs["base_url"] = config.OPENAI_API_BASE
    return ChatOpenAI(**kwargs)


def llm_call_sync(system_prompt: str, user_prompt: str, max_retries: int = 5) -> str:
    """同步调用 LLM，带速率限制和指数退避重试"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    for attempt in range(max_retries + 1):
        _semaphore.acquire()
        try:
            _wait_for_rate_limit()
            llm = get_llm()
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "Too Many Requests" in error_str
            is_server_error = "503" in error_str or "500" in error_str

            if (is_rate_limit or is_server_error) and attempt < max_retries:
                wait = min(2 ** attempt * 2, 30)  # 2s, 4s, 8s, 16s, 30s
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.0fs...",
                    attempt + 1, max_retries, error_str[:80], wait
                )
                time.sleep(wait)
            else:
                logger.error("LLM call failed permanently: %s", error_str[:120])
                raise
        finally:
            _semaphore.release()

    # Should not reach here, but just in case
    raise RuntimeError("LLM call exhausted all retries")


async def llm_call(system_prompt: str, user_prompt: str) -> str:
    """异步调用 LLM（目前包装同步版本以复用速率限制）"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, llm_call_sync, system_prompt, user_prompt
    )
