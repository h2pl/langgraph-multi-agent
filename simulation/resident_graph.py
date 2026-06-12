"""居民独立 Sub-graph — 每个居民拥有自己的 perceive → plan → act 工作流

这是真·多 Agent 架构的核心：每个居民是一个独立的 Agent，
拥有自己的 StateGraph，可以独立感知、规划和行动。
Supervisor Graph 负责编排和协调所有居民的 Sub-graph。
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.resident import Resident
from town.environment import Town
from simulation.interactions import generate_plan, rate_importance


# ─── 居民子图状态 ──────────────────────────────────────────────

class ResidentState(TypedDict):
    """单个居民的子图状态"""
    resident_name: str
    time_str: str
    period: str
    # 感知结果
    current_location: str
    nearby_agents: list[str]
    observation: str
    # 计划结果
    planned_location: str
    planned_activity: str
    planned_emotion: str
    # 可选项
    location_options: list[str]
    # 输出事件
    events: list[str]


# ─── 居民 Agent ────────────────────────────────────────────────

class ResidentAgent:
    """每个居民的独立 Agent，拥有自己的 LangGraph Sub-graph。

    Sub-graph 工作流:
        perceive → plan → act → END

    Supervisor 负责：
    1. 为每个 ResidentAgent 准备输入状态
    2. 并行调用所有 ResidentAgent 的 Sub-graph
    3. 收集输出，更新全局状态
    """

    def __init__(self, resident: Resident, town: Town):
        self.resident = resident
        self.town = town
        self.graph = self._build_graph()

    def _build_graph(self) -> object:
        """构建居民的独立子图"""
        graph = StateGraph(ResidentState)

        graph.add_node("perceive", self._perceive)
        graph.add_node("plan", self._plan)
        graph.add_node("act", self._act)

        graph.set_entry_point("perceive")
        graph.add_edge("perceive", "plan")
        graph.add_edge("plan", "act")
        graph.add_edge("act", END)

        return graph.compile()

    def run(self, time_str: str, period: str, location_options: list[str]) -> ResidentState:
        """运行一个完整的 perceive → plan → act 周期"""
        initial_state = ResidentState(
            resident_name=self.resident.name,
            time_str=time_str,
            period=period,
            current_location=self.resident.current_location,
            nearby_agents=self.town.get_nearby_agents(
                self.resident.name, self.resident.current_location
            ),
            observation="",
            planned_location=self.resident.current_location,
            planned_activity=self.resident.current_activity,
            planned_emotion=self.resident.current_emotion,
            location_options=location_options,
            events=[],
        )
        return self.graph.invoke(initial_state)

    def _make_initial_state(self, time_str: str, period: str,
                            location_options: list[str]) -> dict:
        """创建初始状态字典"""
        return dict(
            resident_name=self.resident.name,
            time_str=time_str,
            period=period,
            current_location=self.resident.current_location,
            nearby_agents=self.town.get_nearby_agents(
                self.resident.name, self.resident.current_location
            ),
            observation="",
            planned_location=self.resident.current_location,
            planned_activity=self.resident.current_activity,
            planned_emotion=self.resident.current_emotion,
            location_options=location_options,
            events=[],
        )

    PHASES = ["perceive", "plan", "act"]

    def run_phase(self, phase: str, state: dict) -> dict:
        """运行单个阶段，返回更新后的状态"""
        fn = {"perceive": self._perceive, "plan": self._plan, "act": self._act}[phase]
        result = fn(state)
        return {**state, **result}

    def run_with_callbacks(self, time_str: str, period: str,
                           location_options: list[str],
                           on_phase_done=None,
                           on_phase_start=None) -> dict:
        """逐步执行 perceive → plan → act，每步回调通知

        Args:
            on_phase_done: 可选回调 (agent_name, phase, new_events, agent_state)
                          每个阶段完成后调用。new_events 仅包含该阶段新增的事件。
            on_phase_start: 可选回调 (agent_name, phase)
                           每个阶段开始前调用，用于 UI 显示"正在执行..."。
        """
        state = self._make_initial_state(time_str, period, location_options)

        for phase in self.PHASES:
            if on_phase_start:
                on_phase_start(self.resident.name, phase)

            prev_event_count = len(state.get("events", []))
            state = self.run_phase(phase, state)
            new_events = state.get("events", [])[prev_event_count:]

            if on_phase_done:
                on_phase_done(self.resident.name, phase, new_events, state)

        return state

    # ─── 子图节点 ────────────────────────────────────────────

    def _perceive(self, state: ResidentState) -> dict:
        """感知：观察周围环境"""
        resident = self.resident
        location = resident.current_location
        nearby = self.town.get_nearby_agents(resident.name, location)
        loc_desc = self.town.describe_location(location)

        if nearby:
            obs = f"在{location}，看到了{'、'.join(nearby)}。{loc_desc}"
        else:
            obs = f"在{location}，周围没有其他人。环境: {loc_desc}"

        # 写入记忆
        importance = min(3.0 + len(nearby) * 1.5, 8.0)
        resident.memory.add_observation(obs, importance, state["time_str"])

        return {
            "observation": obs,
            "nearby_agents": nearby,
            "events": [f"[感知] {resident.name}: {obs}"],
        }

    def _plan(self, state: ResidentState) -> dict:
        """规划：决定下一步行动"""
        resident = self.resident

        plan = generate_plan(
            resident,
            state["time_str"],
            state["period"],
            state["location_options"],
        )

        location = plan.get("location", resident.current_location)
        activity = plan.get("activity", "闲逛")
        emotion = plan.get("emotion", "平静")

        # 验证地点合法性
        if location not in state["location_options"]:
            location = resident.current_location

        # 写入记忆
        resident.memory.add_plan(
            f"计划: 去{location}{activity}", state["time_str"]
        )
        resident.current_plan = f"去{location}{activity}"

        return {
            "planned_location": location,
            "planned_activity": activity,
            "planned_emotion": emotion,
            "events": state["events"] + [
                f"[计划] {resident.name}: 去{location}{activity}（心情: {emotion}）"
            ],
        }

    def _act(self, state: ResidentState) -> dict:
        """行动：执行计划（移动和更新状态）"""
        resident = self.resident
        events = list(state["events"])

        target_loc = state["planned_location"]
        old_loc = resident.current_location

        # 移动
        if old_loc != target_loc:
            self.town.move_agent(resident.name, old_loc, target_loc)
            resident.current_location = target_loc
            events.append(f"[移动] {resident.name}: {old_loc} → {target_loc}")

        # 更新状态
        resident.current_activity = state["planned_activity"]
        resident.current_emotion = state["planned_emotion"]
        events.append(f"[行动] {resident.name}: {resident.status_summary}")

        return {
            "current_location": target_loc,
            "events": events,
        }
