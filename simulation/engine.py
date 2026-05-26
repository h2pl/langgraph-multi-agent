"""LangGraph 模拟引擎 - 小镇生活的核心循环

工作流（StateGraph）:
  perceive -> plan -> act -> [check_day_end] -> perceive (继续) / reflect (结束一天)
                                                              |
                                                              v
                                                             END
"""

from __future__ import annotations

import json
import random
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END

from town.environment import Town
from town.clock import SimClock
from agents.resident import Resident
from agents.profiles import RESIDENT_PROFILES
from simulation.interactions import (
    generate_plan,
    generate_conversation,
    generate_reflection,
    rate_importance,
)
from config import config


# ─── State 定义 ───────────────────────────────────────────────

class SimulationState(TypedDict):
    """模拟状态，在 LangGraph 节点之间传递"""
    step: int
    day: int
    hour: int
    time_str: str
    period: str
    agent_states: dict  # name -> agent state dict
    location_states: dict  # location -> occupants
    events: list[str]  # 本轮事件
    day_log: list[str]  # 当天所有事件
    conversations: list[dict]  # 本轮对话
    should_end_day: bool


# ─── 模拟引擎 ────────────────────────────────────────────────

class TownSimulation:
    """小镇模拟引擎"""

    def __init__(self):
        self.town = Town()
        self.clock = SimClock()
        self.residents: dict[str, Resident] = {}
        self.graph = None
        self.all_events: list[dict] = []  # 所有历史事件

        self._init_residents()
        self._build_graph()

    def _init_residents(self) -> None:
        """初始化所有居民"""
        for profile in RESIDENT_PROFILES:
            resident = Resident.from_profile(profile)
            self.residents[resident.name] = resident
            # 将居民放到初始位置
            self.town.move_agent(resident.name, "", resident.home)

    def _build_graph(self) -> None:
        """构建 LangGraph 工作流"""
        graph = StateGraph(SimulationState)

        # 添加节点
        graph.add_node("perceive", self._perceive_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("act", self._act_node)
        graph.add_node("reflect", self._reflect_node)

        # 设置入口
        graph.set_entry_point("perceive")

        # 添加边
        graph.add_edge("perceive", "plan")
        graph.add_edge("plan", "act")
        graph.add_conditional_edges(
            "act",
            self._should_end_day,
            {
                "continue": "perceive",
                "reflect": "reflect",
            },
        )
        graph.add_edge("reflect", END)

        self.graph = graph.compile()

    # ─── LangGraph 节点 ──────────────────────────────────────

    def _perceive_node(self, state: SimulationState) -> dict:
        """感知节点：每个居民观察周围环境"""
        events = []

        for name, resident in self.residents.items():
            location = resident.current_location
            nearby = self.town.get_nearby_agents(name, location)
            loc_desc = self.town.describe_location(location)

            # 构造观察
            if nearby:
                obs = f"在{location}，看到了{', '.join(nearby)}。{loc_desc}"
            else:
                obs = f"在{location}，周围没有其他人。环境: {loc_desc}"

            importance = min(3.0 + len(nearby) * 1.5, 8.0)
            resident.memory.add_observation(obs, importance, self.clock.time_str)
            events.append(f"[感知] {name}: {obs}")

        return {"events": events}

    def _plan_node(self, state: SimulationState) -> dict:
        """计划节点：每个居民决定下一步行动"""
        events = []
        all_locations = self.town.get_location_names()

        for name, resident in self.residents.items():
            plan = generate_plan(
                resident,
                self.clock.time_str,
                self.clock.period,
                all_locations,
            )

            location = plan.get("location", resident.current_location)
            activity = plan.get("activity", "闲逛")
            emotion = plan.get("emotion", "平静")

            # 验证地点
            if location not in all_locations:
                location = resident.current_location

            resident.current_plan = f"去{location}{activity}"
            events.append(f"[计划] {name}: 打算去{location}{activity}（心情: {emotion}）")

            # 保存计划到记忆
            resident.memory.add_plan(
                f"计划: 去{location}{activity}", self.clock.time_str
            )

        return {"events": state["events"] + events}

    def _act_node(self, state: SimulationState) -> dict:
        """行动节点：执行计划，处理交互"""
        events = []
        conversations = []

        # 1. 执行移动和活动
        for name, resident in self.residents.items():
            plan = resident.current_plan
            # 从计划中提取目标地点
            for loc_name in self.town.get_location_names():
                if loc_name in plan:
                    old_loc = resident.current_location
                    if old_loc != loc_name:
                        self.town.move_agent(name, old_loc, loc_name)
                        resident.current_location = loc_name
                        events.append(f"[移动] {name}: {old_loc} → {loc_name}")
                    break

            # 更新活动和情绪
            if "activity" in plan:
                resident.current_activity = plan
            events.append(f"[行动] {name}: {resident.status_summary}")

        # 2. 处理同一地点的居民交互
        interaction_pairs = set()
        for loc_name in self.town.get_location_names():
            agents_here = self.town.get_agents_at(loc_name)
            if len(agents_here) >= 2:
                # 随机选择一对进行对话（避免太多 LLM 调用）
                pair = tuple(sorted(random.sample(agents_here, 2)))
                if pair not in interaction_pairs:
                    interaction_pairs.add(pair)
                    a1 = self.residents[pair[0]]
                    a2 = self.residents[pair[1]]

                    context = (
                        f"时间: {self.clock.time_str}，地点: {loc_name}，"
                        f"{a1.name}正在{a1.current_activity}，"
                        f"{a2.name}正在{a2.current_activity}"
                    )

                    conversation = generate_conversation(a1, a2, context)
                    conversations.append({
                        "time": self.clock.time_str,
                        "location": loc_name,
                        "participants": list(pair),
                        "content": conversation,
                    })

                    # 将对话加入双方记忆
                    conv_summary = f"在{loc_name}和{a2.name}聊了天"
                    importance = rate_importance(conv_summary, a1)
                    a1.memory.add_observation(
                        f"和{a2.name}在{loc_name}聊天: {conversation[:100]}...",
                        importance,
                        self.clock.time_str,
                    )
                    a2.memory.add_observation(
                        f"和{a1.name}在{loc_name}聊天: {conversation[:100]}...",
                        importance,
                        self.clock.time_str,
                    )

                    events.append(f"[对话] {pair[0]}和{pair[1]}在{loc_name}聊天")

        # 3. 推进时间
        self.clock.tick()

        # 记录事件
        for event in events:
            self.town.add_event(self.clock.time_str, event)

        return {
            "step": state["step"] + 1,
            "day": self.clock.day,
            "hour": self.clock.hour,
            "time_str": self.clock.time_str,
            "period": self.clock.period,
            "events": state["events"] + events,
            "day_log": state["day_log"] + events,
            "conversations": state["conversations"] + conversations,
            "should_end_day": self.clock.is_day_end,
            "agent_states": {n: r.to_dict() for n, r in self.residents.items()},
            "location_states": {
                loc: self.town.get_agents_at(loc)
                for loc in self.town.get_location_names()
            },
        }

    def _reflect_node(self, state: SimulationState) -> dict:
        """反思节点：一天结束时，居民反思当天的经历"""
        events = []

        for name, resident in self.residents.items():
            if resident.memory.should_reflect(config.REFLECTION_THRESHOLD):
                reflection = generate_reflection(resident, self.clock.time_str)
                importance = rate_importance(reflection, resident)
                resident.memory.add_reflection(reflection, importance, self.clock.time_str)
                events.append(f"[反思] {name}: {reflection}")

            # 回家
            old_loc = resident.current_location
            if old_loc != resident.home:
                self.town.move_agent(name, old_loc, resident.home)
                resident.current_location = resident.home
            resident.current_activity = "睡觉"
            resident.current_emotion = "平静"

        return {
            "events": state["events"] + events,
            "day_log": state["day_log"] + events,
            "agent_states": {n: r.to_dict() for n, r in self.residents.items()},
        }

    # ─── 条件路由 ─────────────────────────────────────────────

    def _should_end_day(self, state: SimulationState) -> str:
        """判断是否结束当天"""
        if state.get("should_end_day", False):
            return "reflect"
        if state["step"] >= config.MAX_STEPS_PER_DAY:
            return "reflect"
        return "continue"

    # ─── 公共接口 ─────────────────────────────────────────────

    def get_initial_state(self) -> SimulationState:
        """获取初始状态"""
        return SimulationState(
            step=0,
            day=self.clock.day,
            hour=self.clock.hour,
            time_str=self.clock.time_str,
            period=self.clock.period,
            agent_states={n: r.to_dict() for n, r in self.residents.items()},
            location_states={
                loc: self.town.get_agents_at(loc)
                for loc in self.town.get_location_names()
            },
            events=[],
            day_log=[],
            conversations=[],
            should_end_day=False,
        )

    def run_day(self) -> SimulationState:
        """运行一整天的模拟"""
        initial_state = self.get_initial_state()
        final_state = self.graph.invoke(initial_state)

        # 保存历史
        self.all_events.extend(
            {"day": self.clock.day, "event": e} for e in final_state["day_log"]
        )

        return final_state

    def run_step(self) -> dict:
        """运行单步（用于 Web 实时展示）"""
        state = self.get_initial_state()
        state["step"] = 0

        # 手动执行一个 perceive -> plan -> act 周期
        state = {**state, **self._perceive_node(state)}
        state = {**state, **self._plan_node(state)}
        state = {**state, **self._act_node(state)}

        return state

    def get_status(self) -> dict:
        """获取当前模拟状态"""
        return {
            "clock": self.clock.to_dict(),
            "town": self.town.to_dict(),
            "residents": {n: r.to_dict() for n, r in self.residents.items()},
        }
