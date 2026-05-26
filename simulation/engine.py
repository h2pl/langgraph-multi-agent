"""LangGraph 模拟引擎 - Supervisor + Sub-graph 架构

架构设计：
  Supervisor Graph（全局编排）:
    dispatch_agents → interact → advance_time → [check_day_end]
                                                   ├─ continue → dispatch_agents
                                                   └─ end_day  → reflect → END

  每个居民拥有独立的 Sub-graph（ResidentAgent）:
    perceive → plan → act → END

  Supervisor 负责：编排时间、分发任务、协调交互
  Sub-graph 负责：独立感知、规划、行动
"""

from __future__ import annotations

import random
from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END

from town.environment import Town
from town.clock import SimClock
from agents.resident import Resident
from agents.profiles import RESIDENT_PROFILES
from simulation.resident_graph import ResidentAgent
from simulation.interactions import (
    generate_conversation,
    generate_reflection,
    rate_importance,
)
from config import config


# ─── Supervisor State ────────────────────────────────────────

class SimulationState(TypedDict):
    """Supervisor 全局状态"""
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


# ─── Supervisor 模拟引擎 ─────────────────────────────────────

class TownSimulation:
    """Supervisor 模式的小镇模拟引擎

    架构说明：
    - 每个居民是一个独立的 ResidentAgent，拥有自己的 Sub-graph
    - Supervisor Graph 负责全局编排（时间、交互、反思）
    - dispatch_agents 节点并行调用所有居民的 Sub-graph
    """

    def __init__(self):
        self.town = Town()
        self.clock = SimClock()
        self.residents: dict[str, Resident] = {}
        self.agents: dict[str, ResidentAgent] = {}  # 每个居民的独立 Agent
        self.graph = None
        self.all_events: list[dict] = []

        self._init_residents()
        self._build_supervisor_graph()

    def _init_residents(self) -> None:
        """初始化居民和对应的独立 Agent"""
        for profile in RESIDENT_PROFILES:
            resident = Resident.from_profile(profile)
            self.residents[resident.name] = resident
            self.town.move_agent(resident.name, "", resident.home)

            # 为每个居民创建独立的 Sub-graph Agent
            self.agents[resident.name] = ResidentAgent(resident, self.town)

    def _build_supervisor_graph(self) -> None:
        """构建 Supervisor Graph

        Supervisor 工作流:
          dispatch_agents → interact → advance_time
               ↑                          │
               │      (day not ended)      │
               └───────────────────────────┤
                                           │ (day ended)
                                           ↓
                                        reflect → END
        """
        graph = StateGraph(SimulationState)

        # Supervisor 节点
        graph.add_node("dispatch_agents", self._dispatch_agents_node)
        graph.add_node("interact", self._interact_node)
        graph.add_node("advance_time", self._advance_time_node)
        graph.add_node("reflect", self._reflect_node)

        # 编排流程
        graph.set_entry_point("dispatch_agents")
        graph.add_edge("dispatch_agents", "interact")
        graph.add_edge("interact", "advance_time")
        graph.add_conditional_edges(
            "advance_time",
            self._should_end_day,
            {
                "continue": "dispatch_agents",
                "end_day": "reflect",
            },
        )
        graph.add_edge("reflect", END)

        self.graph = graph.compile()

    # ─── Supervisor 节点 ─────────────────────────────────────

    def _dispatch_agents_node(self, state: SimulationState) -> dict:
        """Supervisor 核心：并行调度所有居民的独立 Sub-graph

        每个 ResidentAgent 的 Sub-graph 会独立执行:
          perceive → plan → act
        Supervisor 收集所有结果并合并。
        """
        all_events = []
        location_options = self.town.get_location_names()

        # 并行运行所有居民的 Sub-graph
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(
                    agent.run,
                    self.clock.time_str,
                    self.clock.period,
                    location_options,
                ): name
                for name, agent in self.agents.items()
            }

            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result()
                    all_events.extend(result.get("events", []))
                except Exception as e:
                    all_events.append(f"[错误] {agent_name} 执行失败: {e}")

        return {
            "events": state["events"] + all_events,
            "day_log": state["day_log"] + all_events,
            "agent_states": {n: r.to_dict() for n, r in self.residents.items()},
            "location_states": {
                loc: self.town.get_agents_at(loc)
                for loc in self.town.get_location_names()
            },
        }

    def _interact_node(self, state: SimulationState) -> dict:
        """Supervisor 交互节点：处理同一地点居民的对话

        只有 Supervisor 知道全局位置信息，
        所以由 Supervisor 负责协调居民之间的交互。
        """
        events = []
        conversations = []

        interaction_pairs = set()
        for loc_name in self.town.get_location_names():
            agents_here = self.town.get_agents_at(loc_name)
            if len(agents_here) >= 2:
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

                    # 写入双方记忆
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

        return {
            "events": state["events"] + events,
            "day_log": state["day_log"] + events,
            "conversations": state["conversations"] + conversations,
        }

    def _advance_time_node(self, state: SimulationState) -> dict:
        """Supervisor 时间节点：推进模拟时钟"""
        self.clock.tick()

        # 记录事件到小镇日志
        for event in state["events"]:
            self.town.add_event(self.clock.time_str, event)

        return {
            "step": state["step"] + 1,
            "day": self.clock.day,
            "hour": self.clock.hour,
            "time_str": self.clock.time_str,
            "period": self.clock.period,
            "should_end_day": self.clock.is_day_end,
            "agent_states": {n: r.to_dict() for n, r in self.residents.items()},
            "location_states": {
                loc: self.town.get_agents_at(loc)
                for loc in self.town.get_location_names()
            },
        }

    def _reflect_node(self, state: SimulationState) -> dict:
        """Supervisor 反思节点：一天结束，协调所有居民反思"""
        events = []

        # 并行执行反思
        with ThreadPoolExecutor(max_workers=len(self.residents)) as executor:
            futures = {}
            for name, resident in self.residents.items():
                if resident.memory.should_reflect(config.REFLECTION_THRESHOLD):
                    futures[executor.submit(
                        generate_reflection, resident, self.clock.time_str
                    )] = (name, resident)

            for future in as_completed(futures):
                name, resident = futures[future]
                try:
                    reflection = future.result()
                    importance = rate_importance(reflection, resident)
                    resident.memory.add_reflection(
                        reflection, importance, self.clock.time_str
                    )
                    events.append(f"[反思] {name}: {reflection}")
                except Exception as e:
                    events.append(f"[错误] {name} 反思失败: {e}")

        # 所有居民回家睡觉
        for name, resident in self.residents.items():
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
        """Supervisor 判断是否结束当天"""
        if state.get("should_end_day", False):
            return "end_day"
        if state["step"] >= config.MAX_STEPS_PER_DAY:
            return "end_day"
        return "continue"

    # ─── 公共接口（保持与 Web 层兼容）───────────────────────────

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

        self.all_events.extend(
            {"day": self.clock.day, "event": e} for e in final_state["day_log"]
        )

        return final_state

    def run_step(self) -> dict:
        """运行单步（用于 Web 实时展示）"""
        state = self.get_initial_state()

        # 单步：dispatch → interact → advance_time
        state = {**state, **self._dispatch_agents_node(state)}
        state = {**state, **self._interact_node(state)}
        state = {**state, **self._advance_time_node(state)}

        return state

    def get_status(self) -> dict:
        """获取当前模拟状态"""
        return {
            "clock": self.clock.to_dict(),
            "town": self.town.to_dict(),
            "residents": {n: r.to_dict() for n, r in self.residents.items()},
        }
