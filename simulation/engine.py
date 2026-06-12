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
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END

from town.environment import Town
logger = logging.getLogger(__name__)
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


# ─── 事件格式化工具 ─────────────────────────────────────────

def _format_event(time_str: str, content: str) -> str:
    """为事件统一加模拟时间戳：[hh:00] 原内容"""
    return f"[{time_str.split(' ', 1)[-1]}] {content}"


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

        # 微步状态：跟踪当前小时内各居民的 perceive/plan/act 进度
        self._micro_state: dict | None = None  # 当前微步上下文
        self._micro_agent_idx: int = 0         # 当前居民序号
        self._micro_phase_idx: int = 0         # 当前阶段序号 (0=perceive,1=plan,2=act)
        self._micro_hour_phase: str = "dispatch"  # dispatch / interact / advance / done

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
        mode = "LLM (大模型生成)" if config.USE_LLM else "Random (系统随机模拟)"
        logger.info("Supervisor graph compiled | Mode: %s", mode)

    # ─── Supervisor 节点 ─────────────────────────────────────

    def _dispatch_agents_node(self, state: SimulationState,
                              on_micro_step=None,
                              on_micro_step_start=None) -> dict:
        """Supervisor 核心：调度所有居民的独立 Sub-graph

        每个 ResidentAgent 的 Sub-graph 会独立执行:
          perceive → plan → act
        Supervisor 收集所有结果并合并。

        Args:
            on_micro_step: 可选回调 (agent_name, phase, stamped_new_events,
                           agent_dict, location_states)
                           每个居民的每个阶段完成后立即调用，用于实时推送。
                           提供此回调时居民串行执行以保证推送顺序。
            on_micro_step_start: 可选回调 (agent_name, phase)
                                 每个阶段开始前调用，用于 UI 显示"正在执行..."。
        """
        all_events = []
        logger.info("Dispatching %d agents", len(self.agents))
        location_options = self.town.get_location_names()

        if on_micro_step:
            # 串行逐步执行：每个居民的每个阶段都回调
            for name, agent in self.agents.items():
                try:
                    def _make_phase_cb(agent_name_bound):
                        def _phase_cb(agent_name, phase, new_events, agent_state):
                            stamped = [_format_event(self.clock.time_str, e)
                                       for e in new_events]
                            on_micro_step(
                                agent_name, phase, stamped,
                                self.residents[agent_name].to_dict(),
                                {loc: self.town.get_agents_at(loc)
                                 for loc in self.town.get_location_names()},
                            )
                        return _phase_cb

                    def _make_start_cb(agent_name_bound):
                        def _start_cb(agent_name, phase):
                            if on_micro_step_start:
                                on_micro_step_start(agent_name, phase)
                        return _start_cb

                    logger.debug("Running agent %s with callbacks", name)
                    result = agent.run_with_callbacks(
                        self.clock.time_str, self.clock.period,
                        location_options,
                        on_phase_done=_make_phase_cb(name),
                        on_phase_start=_make_start_cb(name),
                    )
                    agent_events = result.get("events", [])
                    stamped_agent = [_format_event(self.clock.time_str, e)
                                     for e in agent_events]
                    all_events.extend(stamped_agent)
                    logger.debug("Agent %s completed with %d events", name, len(agent_events))
                except Exception as e:
                    err = _format_event(self.clock.time_str,
                                        f"[错误] {name} 执行失败: {e}")
                    all_events.append(err)
                    logger.error("Agent %s failed: %s", name, e, exc_info=True)
        else:
            # 并行执行（无实时推送需求时更快）
            max_workers = max(2, len(self.agents))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                        agent_events = result.get("events", [])
                        stamped = [_format_event(self.clock.time_str, e)
                                   for e in agent_events]
                        all_events.extend(stamped)
                    except Exception as e:
                        err = _format_event(self.clock.time_str,
                                            f"[错误] {agent_name} 执行失败: {e}")
                        all_events.append(err)
                        logger.error("Agent %s failed: %s", agent_name, e)

        return {
            "events": state["events"] + all_events,
            "day_log": state["day_log"] + all_events,
            "agent_states": {n: r.to_dict() for n, r in self.residents.items()},
            "location_states": {
                loc: self.town.get_agents_at(loc)
                for loc in self.town.get_location_names()
            },
        }

    def _interact_node(self, state: SimulationState,
                       on_conversation_done=None) -> dict:
        logger.info("Starting interaction node")
        """Supervisor 交互节点：处理同一地点居民的对话

        只有 Supervisor 知道全局位置信息，
        所以由 Supervisor 负责协调居民之间的交互。

        Args:
            on_conversation_done: 可选回调 (event_text, conversation_dict)
                                  每段对话完成后立即调用，用于实时推送。
        """
        events = []
        conversations = []

        # 本轮已经配对的居民，避免同一轮内被多次拉去对话
        paired_this_round: set[str] = set()
        for loc_name in self.town.get_location_names():
            agents_here = [
                a for a in self.town.get_agents_at(loc_name)
                if a not in paired_this_round
            ]
            if len(agents_here) < 2:
                continue
            # 随机打乱，再依次两两配对，确保每个未配对的居民都有机会参与
            random.shuffle(agents_here)
            for i in range(0, len(agents_here) - 1, 2):
                a1_name, a2_name = agents_here[i], agents_here[i + 1]
                a1 = self.residents[a1_name]
                a2 = self.residents[a2_name]
                paired_this_round.update([a1_name, a2_name])

                context = (
                    f"时间: {self.clock.time_str}，地点: {loc_name}，"
                    f"{a1.name}正在{a1.current_activity}，"
                    f"{a2.name}正在{a2.current_activity}"
                )

                conversation = generate_conversation(a1, a2, context)
                conv_dict = {
                    "time": self.clock.time_str,
                    "location": loc_name,
                    "participants": [a1_name, a2_name],
                    "content": conversation,
                }
                conversations.append(conv_dict)

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

                conv_tag = "[对话](已进行RAG检索)" if config.USE_LLM else "[对话]"
                event_text = _format_event(
                    self.clock.time_str,
                    f"{conv_tag} {a1_name}和{a2_name}在{loc_name}聊天（已写入记忆）",
                )
                events.append(event_text)
                logger.debug("Conversation between %s and %s at %s", a1_name, a2_name, loc_name)

                if on_conversation_done:
                    on_conversation_done(event_text, conv_dict)

        logger.info("Interaction node produced %d events", len(events))
        # 事件已经在生成时直接加了时间戳
        return {
            "events": state["events"] + events,
            "day_log": state["day_log"] + events,
            "conversations": state["conversations"] + conversations,
        }

    def _advance_time_node(self, state: SimulationState) -> dict:
        logger.info("Advancing time node")
        """Supervisor 时间节点：推进模拟时钟"""
        self.clock.tick()

        # 记录事件到小镇日志
        for event in state["events"]:
            self.town.add_event(self.clock.time_str, event)

        logger.info("Time advanced to %s (%s)", self.clock.time_str, self.clock.period)
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
        logger.info("Starting reflection node")
        """Supervisor 反思节点：一天结束，协调所有居民反思"""
        events = []
        # 翻页前记录时间戳，确保反思事件显示为当天 22:00
        reflect_time_str = self.clock.time_str

        # 并行执行反思（线程数放宽，真正的限流交给 llm_utils 的信号量）
        max_workers = max(2, len(self.residents))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for name, resident in self.residents.items():
                if resident.memory.should_reflect(config.REFLECTION_THRESHOLD):
                futures[executor.submit(
                    generate_reflection, resident, reflect_time_str
                )] = (name, resident)

            for future in as_completed(futures):
                name, resident = futures[future]
                try:
                    reflection = future.result()
                    importance = rate_importance(reflection, resident)
                    resident.memory.add_reflection(
                        reflection, importance, reflect_time_str
                    )
                    ref_tag = "[反思](已进行RAG检索)" if config.USE_LLM else "[反思]"
                    events.append(_format_event(
                        reflect_time_str,
                        f"{ref_tag} {name}: {reflection}（已写入记忆）",
                    ))
                    logger.debug("Reflection by %s: %s", name, reflection)
                except Exception as e:
                    events.append(_format_event(
                        reflect_time_str,
                        f"[错误] {name} 反思失败: {e}",
                    ))

        # 所有居民回家睡觉
        for name, resident in self.residents.items():
            old_loc = resident.current_location
            if old_loc != resident.home:
                self.town.move_agent(name, old_loc, resident.home)
                resident.current_location = resident.home
            resident.current_activity = "睡觉"
            resident.current_emotion = "平静"

        # 反思完成后翻页到下一天
        self.clock.advance_to_next_day()

        return {
            "events": state["events"] + events,
            "day_log": state["day_log"] + events,
            "should_end_day": False,
            "agent_states": {n: r.to_dict() for n, r in self.residents.items()},
            "time_str": self.clock.time_str,
            "period": self.clock.period,
            "day": self.clock.day,
            "hour": self.clock.hour,
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
        self._reset_micro()  # 跳过微步，直接执行完整一小时
        state = self.get_initial_state()

        # 如果已经到达 22:00（day_end），单步时直接走 reflect + 翻页
        if self.clock.is_day_end:
            state = {**state, **self._reflect_node(state)}
            return state

        # 单步：dispatch → interact → advance_time
        state = {**state, **self._dispatch_agents_node(state)}
        state = {**state, **self._interact_node(state)}
        state = {**state, **self._advance_time_node(state)}

        return state

    def _reset_micro(self):
        """重置微步状态，准备新的一小时"""
        self._micro_state = None
        self._micro_agent_idx = 0
        self._micro_phase_idx = 0
        self._micro_hour_phase = "dispatch"

    def get_micro_info(self) -> dict:
        """获取当前微步进度信息"""
        agent_names = list(self.agents.keys())
        phases = ResidentAgent.PHASES
        # 22:00 这一步的特殊处理：先反思再翻页
        if self.clock.is_day_end:
            return {
                "hour_phase": "reflect",
                "agent_idx": 0,
                "agent_count": len(agent_names),
                "agent_name": None,
                "phase_idx": 0,
                "phase_name": "reflect",
                "sim_time": self.clock.time_str,
                "sim_period": self.clock.period,
            }
        if self._micro_hour_phase == "dispatch":
            if self._micro_agent_idx < len(agent_names):
                current_agent = agent_names[self._micro_agent_idx]
                current_phase = phases[self._micro_phase_idx] if self._micro_phase_idx < len(phases) else "done"
            else:
                # 所有居民 dispatch 即将完成，下一步是交互
                current_agent = None
                current_phase = "dispatch_finishing"
        else:
            current_agent = None
            current_phase = self._micro_hour_phase
        return {
            "hour_phase": self._micro_hour_phase,
            "agent_idx": self._micro_agent_idx,
            "agent_count": len(agent_names),
            "agent_name": current_agent,
            "phase_idx": self._micro_phase_idx,
            "phase_name": current_phase,
            "sim_time": self.clock.time_str,
            "sim_period": self.clock.period,
        }

    def micro_step(self) -> dict:
        """执行一个微步：当前居民的下一个 perceive/plan/act 阶段

        返回 dict 包含:
            agent_name, phase, events (仅本步新增), agent_state, location_states,
            micro_info (进度信息), finished_hour (本小时是否全部完成)
        """
        agent_names = list(self.agents.keys())
        phases = ResidentAgent.PHASES
        location_options = self.town.get_location_names()

        # 如果已到 22:00，执行反思
        if self.clock.is_day_end:
            state = self.get_initial_state()
            result = self._reflect_node(state)
            state = {**state, **result}
            self._reset_micro()
            return {
                "agent_name": None,
                "phase": "reflect",
                "events": state.get("events", []),
                "agent_state": None,
                "location_states": {loc: self.town.get_agents_at(loc)
                                    for loc in location_options},
                "agent_states": {n: r.to_dict() for n, r in self.residents.items()},
                "micro_info": self.get_micro_info(),
                "finished_hour": True,
            }

        # 阶段: dispatch（逐居民逐阶段）
        if self._micro_hour_phase == "dispatch":
            if self._micro_agent_idx >= len(agent_names):
                # 所有居民 dispatch 完成，进入 interact
                self._micro_hour_phase = "interact"
                return self.micro_step()  # 递归进入下一阶段

            agent_name = agent_names[self._micro_agent_idx]
            agent = self.agents[agent_name]

            # 初始化该居民的子图状态
            if self._micro_phase_idx == 0:
                self._micro_state = agent._make_initial_state(
                    self.clock.time_str, self.clock.period, location_options
                )

            phase = phases[self._micro_phase_idx]
            prev_count = len(self._micro_state.get("events", []))
            self._micro_state = agent.run_phase(phase, self._micro_state)
            new_events = self._micro_state.get("events", [])[prev_count:]

            # 加时间戳
            stamped = [_format_event(self.clock.time_str, e) for e in new_events]

            result = {
                "agent_name": agent_name,
                "phase": phase,
                "events": stamped,
                "agent_state": self.residents[agent_name].to_dict(),
                "location_states": {loc: self.town.get_agents_at(loc)
                                    for loc in location_options},
                "agent_states": None,
                "micro_info": None,
                "finished_hour": False,
            }

            # 推进到下一阶段
            self._micro_phase_idx += 1
            if self._micro_phase_idx >= len(phases):
                # 该居民完成，下一个居民
                self._micro_agent_idx += 1
                self._micro_phase_idx = 0
                self._micro_state = None

            result["micro_info"] = self.get_micro_info()
            return result

        # 阶段: interact
        if self._micro_hour_phase == "interact":
            state = self.get_initial_state()
            interact_result = self._interact_node(state)
            events = interact_result.get("events", [])[len(state["events"]):]
            conversations = interact_result.get("conversations", [])[len(state.get("conversations", [])):]
            self._micro_hour_phase = "advance"
            return {
                "agent_name": None,
                "phase": "interact",
                "events": events,
                "conversations": conversations,
                "agent_state": None,
                "location_states": {loc: self.town.get_agents_at(loc)
                                    for loc in location_options},
                "agent_states": None,
                "micro_info": self.get_micro_info(),
                "finished_hour": False,
            }

        # 阶段: advance
        if self._micro_hour_phase == "advance":
            state = self.get_initial_state()
            advance_result = self._advance_time_node(state)
            events = advance_result.get("events", [])[len(state["events"]):]
            self._reset_micro()
            return {
                "agent_name": None,
                "phase": "advance",
                "events": events,
                "agent_state": None,
                "location_states": {loc: self.town.get_agents_at(loc)
                                    for loc in location_options},
                "agent_states": {n: r.to_dict() for n, r in self.residents.items()},
                "micro_info": self.get_micro_info(),
                "finished_hour": True,
            }

        # fallback
        self._reset_micro()
        return {
            "agent_name": None, "phase": "unknown", "events": [],
            "agent_state": None, "location_states": {}, "agent_states": None,
            "micro_info": self.get_micro_info(), "finished_hour": True,
        }

    def set_time(self, day: int, hour: int) -> None:
        """将模拟时钟调到指定整点（7~22），并重置微步状态"""
        old_time = self.clock.time_str
        self.clock.set_time(day, hour)
        new_time = self.clock.time_str
        self._reset_micro()
        # 时钟回拨时，清除"未来"的事件和记忆
        if new_time != old_time:
            self._cleanup_after_settime(new_time)

    def _cleanup_after_settime(self, target_time: str) -> None:
        """清除时钟拨回后的所有事件和记忆（保留 <= target_time 的数据）"""
        import re
        def _parse(ts: str):
            m = re.match(r'第(\d+)天\s+(\d+):00', ts)
            return (int(m.group(1)), int(m.group(2))) if m else (999999, 0)
        target = _parse(target_time)

        # 清除小镇事件日志
        self.town.event_log = [
            e for e in self.town.event_log
            if _parse(e.get("time", "")) <= target
        ]

        # 清除每个居民的记忆
        for resident in self.residents.values():
            mem = resident.memory
            mem.memories = [
                m for m in mem.memories
                if _parse(getattr(m, "created_at", "")) <= target
            ]

    def end_day(self) -> dict:
        """Web 单步模式下收尾一天：触发 reflect 并翻页"""
        state = self.get_initial_state()
        if not self.clock.is_day_end:
            return {
                **state,
                "events": ["[系统] 还没到一天结束（22:00），无需收尾"],
            }
        return {**state, **self._reflect_node(state)}

    def get_status(self) -> dict:
        """获取当前模拟状态"""
        return {
            "clock": self.clock.to_dict(),
            "town": self.town.to_dict(),
            "residents": {n: r.to_dict() for n, r in self.residents.items()},
        }
