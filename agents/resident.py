"""居民 Agent - 每个居民的状态和行为"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agents.memory import MemoryStream


@dataclass
class Resident:
    """小镇居民"""
    name: str
    age: int
    occupation: str
    personality: str
    daily_routine: str
    home: str
    workplace: str
    relationships: dict[str, str]
    backstory: str

    # 动态状态
    current_location: str = ""
    current_activity: str = "休息"
    current_emotion: str = "平静"
    current_plan: str = ""

    # 记忆系统
    memory: MemoryStream = field(default_factory=MemoryStream)

    def __post_init__(self):
        if not self.current_location:
            self.current_location = self.home
        # 用居民名字初始化记忆流（用于向量存储的 Collection 命名）
        if not self.memory.owner_name:
            self.memory = MemoryStream(owner_name=self.name)

    @property
    def profile_summary(self) -> str:
        """生成角色摘要，用于 LLM prompt"""
        rel_str = "\n".join(f"  - {k}: {v}" for k, v in self.relationships.items())
        return (
            f"姓名: {self.name}\n"
            f"年龄: {self.age}岁\n"
            f"职业: {self.occupation}\n"
            f"性格: {self.personality}\n"
            f"日常作息: {self.daily_routine}\n"
            f"背景故事: {self.backstory}\n"
            f"人际关系:\n{rel_str}"
        )

    @property
    def status_summary(self) -> str:
        """当前状态摘要"""
        return (
            f"{self.name} 正在 {self.current_location}，"
            f"正在{self.current_activity}，心情{self.current_emotion}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "age": self.age,
            "occupation": self.occupation,
            "personality": self.personality,
            "current_location": self.current_location,
            "current_activity": self.current_activity,
            "current_emotion": self.current_emotion,
            "current_plan": self.current_plan,
            "memory": self.memory.to_dict(),
        }

    @classmethod
    def from_profile(cls, profile: dict) -> Resident:
        """从预设档案创建居民"""
        return cls(
            name=profile["name"],
            age=profile["age"],
            occupation=profile["occupation"],
            personality=profile["personality"],
            daily_routine=profile["daily_routine"],
            home=profile["home"],
            workplace=profile["workplace"],
            relationships=profile["relationships"],
            backstory=profile["backstory"],
        )
