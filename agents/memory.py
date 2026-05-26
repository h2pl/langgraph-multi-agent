"""记忆系统 - 观察、反思、计划

参考斯坦福 Generative Agents 论文的三层记忆架构：
1. 观察记忆 (Observation) - 短期记忆，记录所见所闻
2. 反思记忆 (Reflection) - 高层次思考，从多条观察中提炼洞察
3. 计划记忆 (Plan) - 当天的行动计划
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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
        """简单的关键词相关性评分"""
        query_chars = set(query)
        content_chars = set(self.content)
        overlap = len(query_chars & content_chars)
        total = len(query_chars | content_chars)
        return overlap / total if total > 0 else 0

    def total_score(self, query: str = "") -> float:
        """综合得分 = 重要性 + 时近性 + 相关性"""
        importance_w = self.importance / 10.0
        recency_w = self.recency_score
        relevance_w = self.relevance_score(query) if query else 0
        return importance_w + recency_w + relevance_w

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "type": self.memory_type,
            "importance": self.importance,
            "created_at": self.created_at,
        }


class MemoryStream:
    """记忆流 - 管理一个 Agent 的所有记忆"""

    def __init__(self, max_short_term: int = 20):
        self.memories: list[MemoryItem] = []
        self.max_short_term = max_short_term
        self._observation_count_since_reflection = 0

    def add_observation(self, content: str, importance: float, time_str: str) -> None:
        """添加观察记忆"""
        mem = MemoryItem(
            content=content,
            memory_type="observation",
            importance=importance,
            created_at=time_str,
        )
        self.memories.append(mem)
        self._observation_count_since_reflection += 1

    def add_reflection(self, content: str, importance: float, time_str: str) -> None:
        """添加反思记忆"""
        mem = MemoryItem(
            content=content,
            memory_type="reflection",
            importance=max(importance, 7.0),  # 反思至少 7 分重要性
            created_at=time_str,
        )
        self.memories.append(mem)
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

    def should_reflect(self, threshold: int = 5) -> bool:
        """是否应该触发反思"""
        return self._observation_count_since_reflection >= threshold

    def retrieve(self, query: str = "", top_k: int = 10) -> list[MemoryItem]:
        """检索最相关的记忆"""
        for mem in self.memories:
            mem.last_accessed = mem.last_accessed  # 保持原值用于衰减

        scored = sorted(
            self.memories,
            key=lambda m: m.total_score(query),
            reverse=True,
        )
        # 标记被访问
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
            "recent": [m.to_dict() for m in self.memories[-10:]],
        }
