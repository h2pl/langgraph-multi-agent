"""小镇环境 - 定义地点和空间关系"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Location:
    name: str
    description: str
    category: str  # residential, commercial, public, work
    occupants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "occupants": list(self.occupants),
        }


class Town:
    """小镇环境"""

    def __init__(self, name: str = "桃源镇"):
        self.name = name
        self.locations: dict[str, Location] = {}
        self.event_log: list[dict] = []
        self._init_default_locations()

    def _init_default_locations(self) -> None:
        """初始化默认地点"""
        default_locations = [
            Location("老王面馆", "镇上最受欢迎的面馆，老王的手擀面远近闻名", "commercial"),
            Location("小镇广场", "镇中心的广场，居民们喜欢在这里聊天散步", "public"),
            Location("卫生所", "张医生坐诊的地方，为镇上居民看病", "work"),
            Location("桃源小学", "陈老师教书的地方，镇上唯一的小学", "work"),
            Location("赵大姐超市", "什么都卖的小超市，也是八卦集散地", "commercial"),
            Location("镇公园", "有一片小湖和几棵老柳树，适合散步和晨练", "public"),
            Location("老王的家", "老王住的地方，面馆楼上", "residential"),
            Location("小李的家", "小李租的小公寓，堆满了电脑和书", "residential"),
            Location("张医生的家", "张医生和家人住的小院", "residential"),
            Location("陈老师的家", "陈老师的温馨小屋", "residential"),
            Location("赵大姐的家", "超市后面的住所", "residential"),
        ]
        for loc in default_locations:
            self.locations[loc.name] = loc

    def get_location(self, name: str) -> Optional[Location]:
        return self.locations.get(name)

    def move_agent(self, agent_name: str, from_loc: str, to_loc: str) -> None:
        """移动居民到新地点"""
        if from_loc in self.locations:
            loc = self.locations[from_loc]
            if agent_name in loc.occupants:
                loc.occupants.remove(agent_name)

        if to_loc in self.locations:
            loc = self.locations[to_loc]
            if agent_name not in loc.occupants:
                loc.occupants.append(agent_name)

    def get_agents_at(self, location_name: str) -> list[str]:
        """获取某地点的所有居民"""
        loc = self.locations.get(location_name)
        return list(loc.occupants) if loc else []

    def get_nearby_agents(self, agent_name: str, current_location: str) -> list[str]:
        """获取同一地点的其他居民"""
        agents = self.get_agents_at(current_location)
        return [a for a in agents if a != agent_name]

    def add_event(self, time_str: str, event: str) -> None:
        self.event_log.append({"time": time_str, "event": event})

    def get_location_names(self, category: Optional[str] = None) -> list[str]:
        """获取地点名称列表"""
        if category:
            return [name for name, loc in self.locations.items() if loc.category == category]
        return list(self.locations.keys())

    def get_public_locations(self) -> list[str]:
        """获取公共场所"""
        return [
            name for name, loc in self.locations.items()
            if loc.category in ("public", "commercial")
        ]

    def describe_location(self, name: str) -> str:
        """描述一个地点的当前状态"""
        loc = self.locations.get(name)
        if not loc:
            return f"未知地点: {name}"
        occupants_str = "、".join(loc.occupants) if loc.occupants else "空无一人"
        return f"{loc.name}（{loc.description}）- 当前在场: {occupants_str}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "locations": {k: v.to_dict() for k, v in self.locations.items()},
            "event_log": self.event_log[-20:],  # 最近 20 条
        }
