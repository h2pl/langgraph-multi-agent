"""向量存储 - 基于 Chroma 的语义记忆检索

使用 sentence-transformers 本地模型生成 Embedding，
存入 Chroma 向量数据库，实现语义级别的记忆检索。

- 模型: paraphrase-multilingual-MiniLM-L12-v2（支持中文，384维，~120MB）
- 存储: 本地持久化，记忆跨会话保留
- 零 API 费用，全部本地运行
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

# 设置 HuggingFace 国内镜像，加速 SentenceTransformer 模型下载并防止连接超时挂起
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logger = logging.getLogger(__name__)

# Chroma + Embedding 是否可用
_VECTOR_AVAILABLE = False
_chroma_client = None
_embedding_fn = None

# 中文友好的多语言 Embedding 模型
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DB_PATH = str(Path(__file__).parent.parent / "data" / "chroma")

try:
    import chromadb
    from chromadb.utils import embedding_functions

    _VECTOR_AVAILABLE = True
    logger.info("Chroma + sentence-transformers 可用，启用向量语义检索")
except ImportError:
    logger.warning(
        "chromadb 或 sentence-transformers 未安装，回退到文本匹配模式。\n"
        "安装命令: pip install chromadb sentence-transformers"
    )


def is_available() -> bool:
    """向量检索是否可用"""
    from config import config
    return _VECTOR_AVAILABLE and config.USE_VECTOR_STORE


def get_chroma_client() -> Optional[object]:
    """获取 Chroma 客户端（单例）"""
    global _chroma_client
    if not _VECTOR_AVAILABLE:
        return None
    if _chroma_client is None:
        Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _chroma_client


def get_embedding_function() -> Optional[object]:
    """获取 Embedding 函数（单例）

    首次调用时会自动下载模型（约120MB），之后从本地缓存读取。
    """
    global _embedding_fn
    if not _VECTOR_AVAILABLE:
        return None
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_fn


def get_collection(name: str) -> Optional[object]:
    """获取或创建一个 Chroma Collection（每个居民一个）

    Args:
        name: Collection 名称，通常是居民名字
    """
    client = get_chroma_client()
    ef = get_embedding_function()
    if client is None or ef is None:
        return None

    # Chroma collection name 只允许字母数字和下划线
    safe_name = "".join(c if c.isalnum() else "_" for c in name)
    return client.get_or_create_collection(
        name=safe_name,
        embedding_function=ef,
        metadata={"description": f"{name}的记忆向量库"},
    )


def reset_all() -> None:
    """重置所有向量数据（用于模拟重置）"""
    global _chroma_client
    client = get_chroma_client()
    if client:
        for col in client.list_collections():
            client.delete_collection(col.name)
    _chroma_client = None
