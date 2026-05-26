"""记忆系统 - 观察、反思、计划

参考斯坦福 Generative Agents 论文的三层记忆架构：
1. 观察记忆 (Observation) - 短期记忆，记录所见所闻
2. 反思记忆 (Reflection) - 高层次思考，从多条观察中提炼洞察
3. 计划记忆 (Plan) - 当天的行动计划
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional


@dataclass
class MemoryItem:
    """单条记忆"""
    content: str
    memory_type: str  # observation, reflection, plan
    importance: float = 5.0  # 1-10 重要性评分
    created_at: str = ""  # 模拟时间
    last_accessed: float = field(default_factory=time.time)

    @property
    def recency_score(self) -> float:
        """时间衰减分数，越近越高"""
        elapsed = time.time() - self.last_accessed
        decay_factor = 0.995
        return decay_factor ** (elapsed / 60)  # 每分钟衰减

    def relevance_score(self, query: str) -> float:
        """基于序列匹配的文本相似度评分（比字符集合匹配准确得多）"""
        if not query or not self.content:
            return 0.0
        # SequenceMatcher: 找出两段文本的最长公共子序列比率
        seq_score = SequenceMatcher(None, self.content, query).ratio()
        # 关键词重叠: 按词粒度计算 Jaccard 相似度
        query_words = set(query)
        content_words = set(self.content)
        overlap = len(query_words & content_words)
        total = len(query_words | content_words)
        word_score = overlap / total if total > 0 else 0
        # 两种方式加权融合
        return 0.6 * seq_score + 0.4 * word_score

    def total_score(self, query: str = "") -> float:
        """综合得分 = 重要性(α) + 时近性(β) + 相关性(γ)

        权重参考斯坦福论文：三个维度同等重要，各占约 1/3。
        """
        alpha, beta, gamma = 1.0, 1.0, 1.0
        importance_w = self.importance / 10.0
        recency_w = self.recency_score
        relevance_w = self.relevance_score(query) if query else 0
        return alpha * importance_w + beta * recency_w + gamma * relevance_w

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "type": self.memory_type,
            "importance": self.importance,
            "created_at": self.created_at,
        }


class MemoryStream:
    """记忆流 - 管理一个 Agent 的所有记忆

    支持两种检索模式（自动选择）：
    1. 向量语义检索 — Chroma + sentence-transformers（需安装依赖）
    2. 文本匹配检索 — SequenceMatcher（零依赖回退方案）
    """

    def __init__(self, owner_name: str = "", max_short_term: int = 20):
        self.owner_name = owner_name
        self.memories: list[MemoryItem] = []
        self.max_short_term = max_short_term
        self._observation_count_since_reflection = 0
        self._mem_counter = 0  # 用于生成唯一 ID

        # 尝试初始化向量存储
        self._collection = None
        self._init_vector_store()

    def _init_vector_store(self) -> None:
        """初始化 Chroma 向量存储（失败则静默回退）"""
        try:
            from agents.vector_store import is_available, get_collection
            if is_available() and self.owner_name:
                self._collection = get_collection(self.owner_name)
        except Exception:
            self._collection = None

    @property
    def use_vector(self) -> bool:
        """是否使用向量检索"""
        return self._collection is not None

    def _next_id(self) -> str:
        """生成唯一记忆 ID"""
        self._mem_counter += 1
        return f"{self.owner_name}_{self._mem_counter}"

    def _add_to_vector_store(self, mem: MemoryItem) -> None:
        """将记忆同步写入向量数据库"""
        if not self._collection:
            return
        try:
            self._collection.add(
                documents=[mem.content],
                metadatas=[{
                    "type": mem.memory_type,
                    "importance": str(mem.importance),
                    "created_at": mem.created_at,
                }],
                ids=[self._next_id()],
            )
        except Exception:
            pass  # 向量写入失败不影响主流程

    def add_observation(self, content: str, importance: float, time_str: str) -> None:
        """添加观察记忆"""
        mem = MemoryItem(
            content=content,
            memory_type="observation",
            importance=importance,
            created_at=time_str,
        )
        self.memories.append(mem)
        self._add_to_vector_store(mem)
        self._observation_count_since_reflection += 1

    def add_reflection(self, content: str, importance: float, time_str: str) -> None:
        """添加反思记忆"""
        mem = MemoryItem(
            content=content,
            memory_type="reflection",
            importance=max(importance, 7.0),
            created_at=time_str,
        )
        self.memories.append(mem)
        self._add_to_vector_store(mem)
        self._observation_count_since_reflection = 0

    def add_plan(self, content: str, time_str: str) -> None:
        """添加计划"""
        mem = MemoryItem(
            content=content,
            memory_type="plan",
            importance=6.0,
            created_at=time_str,
        )
        self.memories.append(mem)
        self._add_to_vector_store(mem)

    def should_reflect(self, threshold: int = 5) -> bool:
        """是否应该触发反思"""
        return self._observation_count_since_reflection >= threshold

    def retrieve(self, query: str = "", top_k: int = 10) -> list[MemoryItem]:
        """检索最相关的记忆

        向量模式：用 Chroma 语义检索获取 relevance，再结合 importance + recency 重排
        回退模式：用 SequenceMatcher 文本匹配
        """
        if not self.memories:
            return []

        if self.use_vector and query:
            return self._retrieve_vector(query, top_k)
        return self._retrieve_text(query, top_k)

    def _retrieve_vector(self, query: str, top_k: int) -> list[MemoryItem]:
        """向量语义检索（Chroma）"""
        try:
            # 从 Chroma 检索语义相关的记忆（取多一些用于重排）
            n_candidates = min(top_k * 3, len(self.memories))
            results = self._collection.query(
                query_texts=[query],
                n_results=n_candidates,
            )

            if not results or not results["documents"] or not results["documents"][0]:
                return self._retrieve_text(query, top_k)

            # Chroma 返回的距离越小越相似，转为 0~1 的相似度分数
            distances = results["distances"][0] if results.get("distances") else []
            documents = results["documents"][0]

            # 建立 content -> vector_score 的映射
            vector_scores = {}
            for doc, dist in zip(documents, distances):
                # Chroma L2 距离转相似度: similarity = 1 / (1 + distance)
                vector_scores[doc] = 1.0 / (1.0 + dist)

            # 对内存中的记忆重排：综合 向量相似度 + 重要性 + 时近性
            scored_memories = []
            for mem in self.memories:
                vec_score = vector_scores.get(mem.content, 0.0)
                importance_score = mem.importance / 10.0
                recency = mem.recency_score
                # 向量相似度权重更高（这是语义检索的核心优势）
                total = 1.5 * vec_score + 1.0 * importance_score + 1.0 * recency
                scored_memories.append((total, mem))

            scored_memories.sort(key=lambda x: x[0], reverse=True)

            # 标记被访问
            result = [mem for _, mem in scored_memories[:top_k]]
            for mem in result:
                mem.last_accessed = time.time()
            return result

        except Exception:
            # 向量检索失败，回退到文本匹配
            return self._retrieve_text(query, top_k)

    def _retrieve_text(self, query: str, top_k: int) -> list[MemoryItem]:
        """文本匹配检索（回退方案）"""
        scored = sorted(
            self.memories,
            key=lambda m: m.total_score(query),
            reverse=True,
        )
        for mem in scored[:top_k]:
            mem.last_accessed = time.time()
        return scored[:top_k]

    def get_recent_observations(self, n: int = 5) -> list[MemoryItem]:
        """获取最近的观察"""
        obs = [m for m in self.memories if m.memory_type == "observation"]
        return obs[-n:]

    def get_reflections(self) -> list[MemoryItem]:
        """获取所有反思"""
        return [m for m in self.memories if m.memory_type == "reflection"]

    def get_today_plan(self) -> Optional[MemoryItem]:
        """获取最近的计划"""
        plans = [m for m in self.memories if m.memory_type == "plan"]
        return plans[-1] if plans else None

    def format_memories(self, memories: list[MemoryItem]) -> str:
        """格式化记忆为文本"""
        if not memories:
            return "（暂无记忆）"
        lines = []
        for mem in memories:
            type_label = {"observation": "观察", "reflection": "反思", "plan": "计划"}.get(
                mem.memory_type, mem.memory_type
            )
            lines.append(f"[{type_label}][{mem.created_at}] {mem.content}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_memories": len(self.memories),
            "observations": len([m for m in self.memories if m.memory_type == "observation"]),
            "reflections": len([m for m in self.memories if m.memory_type == "reflection"]),
            "vector_enabled": self.use_vector,
            "recent": [m.to_dict() for m in self.memories[-10:]],
        }
