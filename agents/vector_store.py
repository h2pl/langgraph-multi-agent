"""向量存储 - 基于 Chroma 的语义记忆检索

使用 sentence-transformers 本地模型生成 Embedding，
存入 Chroma 向量数据库，实现语义级别的记忆检索。

- 模型: paraphrase-multilingual-MiniLM-L12-v2（支持中文，384维，~120MB）
- 存储: 本地持久化，记忆跨会话保留
- 零 API 费用，全部本地运行
"""

from __future__ import annotations

import hashlib
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

# 中文友好的多语言 Embedding 模型名称
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL_PATH = str(Path(__file__).parent.parent / "data" / "models" / EMBEDDING_MODEL_NAME)
CHROMA_DB_PATH = str(Path(__file__).parent.parent / "data" / "chroma")

try:
    import chromadb
    from chromadb.utils import embedding_functions
    from sentence_transformers import SentenceTransformer

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

    优先从本地 data/models 目录加载。如果本地不存在模型，
    则自动通过镜像源下载并缓存到本地目录，以便后续离线瞬间加载。
    """
    global _embedding_fn
    if not _VECTOR_AVAILABLE:
        return None
    if _embedding_fn is None:
        local_path = Path(LOCAL_MODEL_PATH)
        if not local_path.exists():
            logger.info(f"本地未检测到模型目录: {LOCAL_MODEL_PATH}")
            logger.info(f"正在自动从 HuggingFace 镜像站下载 {EMBEDDING_MODEL_NAME} 并保存到本地项目...")
            try:
                # 临时配置国内镜像站以加速下载
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                
                # 下载并导出模型
                model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                local_path.mkdir(parents=True, exist_ok=True)
                model.save(str(local_path))
                logger.info("模型本地化缓存成功！")
            except Exception as e:
                logger.warning(f"自动下载并缓存模型失败: {e}。将尝试在线/默认缓存加载模式。")

        # 检查是否成功获得本地模型，若有则开启离线模式秒开
        if local_path.exists():
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=str(local_path)
            )
            logger.info("成功加载本地缓存的 Embedding 模型，已切换至离线模式")
        else:
            _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME
            )
            logger.info(f"使用在线/缓存模式加载 Embedding 模型: {EMBEDDING_MODEL_NAME}")

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

    # Chroma collection name 只允许 [a-zA-Z0-9._-]，3-512 字符，必须以字母或数字开头结尾
    # 中文名会转成全下划线导致非法，改用 MD5 哈希确保合法且唯一
    safe_name = "res_" + hashlib.md5(name.encode("utf-8")).hexdigest()[:12]
    return client.get_or_create_collection(
        name=safe_name,
        embedding_function=ef,
        metadata={"owner": name, "description": f"{name}的记忆向量库"},
    )


def reset_all() -> None:
    """重置所有向量数据（用于模拟重置）"""
    global _chroma_client
    client = get_chroma_client()
    if client:
        for col in client.list_collections():
            client.delete_collection(col.name)
    _chroma_client = None
