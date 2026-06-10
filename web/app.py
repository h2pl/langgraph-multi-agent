"""Web 可视化 - FastAPI + WebSocket 实时展示小镇模拟"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from simulation.engine import TownSimulation
from config import config

logger = logging.getLogger(__name__)

app = FastAPI(title="桃源镇 - AI 小镇模拟")

# 静态文件
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 全局模拟实例
simulation: TownSimulation | None = None
connected_clients: list[WebSocket] = []

# 用于取消 run_day 的事件
_cancel_day = asyncio.Event()


def get_simulation() -> TownSimulation:
    global simulation
    if simulation is None:
        simulation = TownSimulation()
    return simulation


async def broadcast_progress(phase: str, detail: str = "", extra: dict | None = None):
    """向所有 WebSocket 客户端广播进度消息"""
    payload: dict = {"phase": phase, "detail": detail}
    if extra:
        payload.update(extra)
    message = json.dumps({"type": "progress", "data": payload}, ensure_ascii=False)

    dead_clients = []
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            dead_clients.append(client)
    for c in dead_clients:
        if c in connected_clients:
            connected_clients.remove(c)


async def broadcast_micro_step(agent_name: str, phase: str, events: list[str],
                               agent_state: dict, location_states: dict):
    """向所有客户端推送单个居民单阶段完成后的实时事件"""
    PHASE_LABELS = {"perceive": "感知", "plan": "计划", "act": "行动",
                    "interact": "交互", "advance": "推进时间", "reflect": "反思"}
    payload = {
        "agent_name": agent_name,
        "phase": phase,
        "phase_label": PHASE_LABELS.get(phase, phase),
        "events": events,
        "agent_state": agent_state,
        "location_states": location_states,
    }
    message = json.dumps({"type": "micro_step", "data": payload}, ensure_ascii=False)
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            pass


async def broadcast_conversation(event_text: str, conversation: dict):
    """向所有客户端推送单段对话完成后的实时事件"""
    payload = {
        "event": event_text,
        "conversation": conversation,
    }
    message = json.dumps({"type": "conversation_event", "data": payload}, ensure_ascii=False)
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            pass


async def run_step_with_progress(sim: TownSimulation) -> dict:
    """运行单步并通过 WebSocket 发送进度（含已用时间）"""
    import time as _time
    sim._reset_micro()  # 完整一小时会覆盖微步进度
    t0 = _time.monotonic()
    state = sim.get_initial_state()

    # 记住本步模拟的时间（tick 前），确保和事件时间戳一致
    step_time = sim.clock.time_str
    step_period = sim.clock.period

    def _elapsed() -> str:
        s = _time.monotonic() - t0
        return f"{s:.1f}s"

    # 如果已到 22:00（day_end），执行反思 + 翻页到下一天
    if sim.clock.is_day_end:
        await broadcast_progress("dispatch", f"正在进行日终反思...",
                                 {"elapsed": "0.0s"})
        state = await asyncio.get_event_loop().run_in_executor(
            None, lambda: {**state, **sim._reflect_node(state)}
        )
        # reflect 后居民已回家、时钟已翻页，刷新位置数据
        state["agent_states"] = {n: r.to_dict() for n, r in sim.residents.items()}
        state["location_states"] = {
            loc: sim.town.get_agents_at(loc)
            for loc in sim.town.get_location_names()
        }
        state["step_time"] = step_time
        state["step_period"] = step_period
        await broadcast_progress("step_done", f"反思完成，进入下一天（耗时 {_elapsed()}）")
        return state

    # 阶段 1: 调度 Agents（逐居民逐阶段实时推送）
    mode_label = "LLM 生成中" if config.USE_LLM else "随机模拟中"
    agent_count = len(sim.agents)
    await broadcast_progress("dispatch", f"正在调度居民行为 ({mode_label})... 0/{agent_count}",
                             {"elapsed": "0.0s"})

    loop = asyncio.get_event_loop()
    micro_queue: asyncio.Queue = asyncio.Queue()
    PHASE_LABELS = {"perceive": "感知", "plan": "计划", "act": "行动"}

    def _on_micro_step(agent_name, phase, stamped_events, agent_dict, loc_states):
        loop.call_soon_threadsafe(micro_queue.put_nowait,
                                  (agent_name, phase, stamped_events, agent_dict, loc_states))

    dispatch_task = asyncio.ensure_future(
        loop.run_in_executor(
            None, lambda: {**state, **sim._dispatch_agents_node(state, on_micro_step=_on_micro_step)}
        )
    )

    done_agents = set()
    while not dispatch_task.done():
        try:
            a_name, a_phase, a_events, a_dict, a_locs = await asyncio.wait_for(
                micro_queue.get(), timeout=0.2)
            await broadcast_micro_step(a_name, a_phase, a_events, a_dict, a_locs)
            done_agents.add(a_name)
            phase_label = PHASE_LABELS.get(a_phase, a_phase)
            await broadcast_progress("dispatch",
                f"{a_name} · {phase_label} ({len(done_agents)}/{agent_count}) · {_elapsed()}",
                {"elapsed": _elapsed()})
        except asyncio.TimeoutError:
            pass

    try:
        state = dispatch_task.result()
    except Exception as exc:
        logger.exception("dispatch_agents failed")
        raise

    # 排空队列中剩余的事件
    while not micro_queue.empty():
        a_name, a_phase, a_events, a_dict, a_locs = micro_queue.get_nowait()
        await broadcast_micro_step(a_name, a_phase, a_events, a_dict, a_locs)
        done_agents.add(a_name)
        phase_label = PHASE_LABELS.get(a_phase, a_phase)
        await broadcast_progress("dispatch",
            f"{a_name} · {phase_label} ({len(done_agents)}/{agent_count}) · {_elapsed()}",
            {"elapsed": _elapsed()})

    # 阶段 2: 交互（逐段对话实时推送）
    await broadcast_progress("interact", f"正在处理居民交互...",
                             {"elapsed": _elapsed()})

    conv_queue: asyncio.Queue = asyncio.Queue()

    def _on_conv_done(event_text, conv_dict):
        loop.call_soon_threadsafe(conv_queue.put_nowait,
                                  (event_text, conv_dict))

    interact_task = asyncio.ensure_future(
        loop.run_in_executor(
            None, lambda: {**state, **sim._interact_node(state, on_conversation_done=_on_conv_done)}
        )
    )

    while not interact_task.done():
        try:
            event_text, conv_dict = await asyncio.wait_for(conv_queue.get(), timeout=0.2)
            await broadcast_conversation(event_text, conv_dict)
        except asyncio.TimeoutError:
            pass

    try:
        state = interact_task.result()
    except Exception as exc:
        logger.exception("interact failed")
        raise

    # 排空队列中剩余的对话
    while not conv_queue.empty():
        event_text, conv_dict = conv_queue.get_nowait()
        await broadcast_conversation(event_text, conv_dict)

    # 阶段 3: 推进时间
    await broadcast_progress("advance", f"正在推进时间...",
                             {"elapsed": _elapsed()})
    state = await asyncio.get_event_loop().run_in_executor(
        None, lambda: {**state, **sim._advance_time_node(state)}
    )

    # 将本步模拟时间写入 state，便于前端显示
    state["step_time"] = step_time
    state["step_period"] = step_period

    await broadcast_progress("step_done", f"步骤完成（耗时 {_elapsed()}）")
    return state


def _build_step_data(state: dict, sim: TownSimulation) -> dict:
    """构建发给前端的 step_update 数据，使用 step_time 保证时间一致性"""
    return {
        "time": state.get("step_time", state["time_str"]),
        "sim_time": state.get("step_time", sim.clock.time_str),
        "sim_period": state.get("step_period", sim.clock.period),
        "sim_day": sim.clock.day,
        "events": state["events"][-10:],
        "conversations": state.get("conversations", []),
        "agent_states": state["agent_states"],
        "location_states": state["location_states"],
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    """主页"""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status():
    """获取当前状态"""
    sim = get_simulation()
    status = sim.get_status()
    status["use_llm"] = config.USE_LLM
    return status


@app.post("/api/step")
async def run_step():
    """执行一步模拟（带进度推送）"""
    sim = get_simulation()
    try:
        state = await run_step_with_progress(sim)
    except Exception as e:
        logger.exception("run_step failed")
        return {"status": "error", "message": str(e)}

    # 通知所有 WebSocket 客户端
    step_data = _build_step_data(state, sim)
    message = json.dumps({"type": "step_update", "data": step_data}, ensure_ascii=False)

    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            pass

    return {"status": "ok", "state": step_data}


@app.post("/api/run_day")
async def run_day():
    """逐步运行一整天（支持暂停）"""
    import time as _time
    sim = get_simulation()
    _cancel_day.clear()
    t0 = _time.monotonic()

    all_events: list[str] = []
    all_conversations: list[dict] = []
    step = 0

    while not sim.clock.is_day_end:
        # 检查是否被暂停
        if _cancel_day.is_set():
            await broadcast_progress("paused", f"已暂停（第 {step} 步，耗时 {_time.monotonic() - t0:.1f}s）")
            return {
                "status": "paused",
                "step": step,
                "events": all_events,
                "conversations": all_conversations,
            }

        step += 1
        elapsed = f"{_time.monotonic() - t0:.1f}s"
        await broadcast_progress("day_running",
            f"第 {step} 步 · {sim.clock.time_str} · 已用时 {elapsed}",
            {"elapsed": elapsed, "step": step, "sim_time": sim.clock.time_str})

        try:
            state = await run_step_with_progress(sim)
        except Exception as e:
            logger.exception("run_day step %d failed", step)
            await broadcast_progress("error", f"第 {step} 步执行失败: {e}")
            break

        all_events.extend(state.get("events", []))
        all_conversations.extend(state.get("conversations", []))

        # 推送本步结果给前端实时更新
        step_data = _build_step_data(state, sim)
        update_msg = json.dumps(
            {"type": "step_update", "data": step_data}, ensure_ascii=False
        )
        for client in connected_clients:
            try:
                await client.send_text(update_msg)
            except Exception:
                pass

        # 短暂延迟，让用户能看到每步的变化（类似自动运行效果）
        await asyncio.sleep(1.0)

    # 到达 22:00，执行反思并翻页到下一天
    await broadcast_progress("day_running",
        f"正在进行日终反思... · 已用时 {_time.monotonic() - t0:.1f}s",
        {"elapsed": f"{_time.monotonic() - t0:.1f}s"})
    reflect_state = sim.get_initial_state()
    reflect_result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: sim._reflect_node(reflect_state)
    )
    reflect_state = {**reflect_state, **reflect_result}
    all_events.extend(reflect_state.get("events", []))

    # 推送反思结果到前端
    reflect_data = {
        "time": sim.clock.time_str,
        "sim_time": sim.clock.time_str,
        "sim_period": sim.clock.period,
        "sim_day": sim.clock.day,
        "events": reflect_state.get("events", [])[-10:],
        "conversations": [],
        "agent_states": {n: r.to_dict() for n, r in sim.residents.items()},
        "location_states": {
            loc: sim.town.get_agents_at(loc)
            for loc in sim.town.get_location_names()
        },
    }
    reflect_msg = json.dumps(
        {"type": "step_update", "data": reflect_data}, ensure_ascii=False
    )
    for client in connected_clients:
        try:
            await client.send_text(reflect_msg)
        except Exception:
            pass

    elapsed = f"{_time.monotonic() - t0:.1f}s"
    await broadcast_progress("done", f"一天模拟完成（耗时 {elapsed}）")
    return {
        "status": "ok",
        "step": step,
        "events": all_events,
        "conversations": all_conversations,
    }


@app.post("/api/micro_step")
async def do_micro_step():
    """执行一个微步（当前居民的下一个 perceive/plan/act 阶段）"""
    sim = get_simulation()
    result = await asyncio.get_event_loop().run_in_executor(
        None, sim.micro_step
    )

    # 实时推送微步结果
    await broadcast_micro_step(
        result.get("agent_name"),
        result["phase"],
        result.get("events", []),
        result.get("agent_state"),
        result.get("location_states", {}),
    )

    # 如果有对话，也推送
    convs = result.get("conversations", [])
    for conv in convs:
        event_text = f"[{sim.clock.time_str.split(' ', 1)[-1]}] [对话] {' & '.join(conv['participants'])} 在{conv['location']}聊天"
        await broadcast_conversation(event_text, conv)

    # 如果完成一小时，刷新全局状态
    if result.get("finished_hour"):
        step_data = {
            "time": sim.clock.time_str,
            "sim_time": sim.clock.time_str,
            "sim_period": sim.clock.period,
            "sim_day": sim.clock.day,
            "events": result.get("events", []),
            "conversations": convs,
            "agent_states": result.get("agent_states") or {n: r.to_dict() for n, r in sim.residents.items()},
            "location_states": result.get("location_states", {}),
        }
        msg = json.dumps({"type": "step_update", "data": step_data}, ensure_ascii=False)
        for client in connected_clients:
            try:
                await client.send_text(msg)
            except Exception:
                pass

    return {"status": "ok", "result": result}


@app.get("/api/micro_info")
async def micro_info():
    """获取当前微步进度"""
    sim = get_simulation()
    return sim.get_micro_info()


@app.post("/api/pause_day")
async def pause_day():
    """暂停正在运行的一天模拟"""
    _cancel_day.set()
    return {"status": "ok", "message": "暂停信号已发送"}


@app.post("/api/reset")
async def reset():
    """重置模拟"""
    global simulation
    await broadcast_progress("reset", "正在重置模拟...")
    simulation = TownSimulation()
    await broadcast_progress("done", "重置完成")
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 实时推送"""
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("action") == "step":
                sim = get_simulation()
                state = await run_step_with_progress(sim)
                step_data = _build_step_data(state, sim)
                await websocket.send_text(json.dumps(
                    {"type": "step_update", "data": step_data},
                    ensure_ascii=False,
                ))

            elif msg.get("action") == "status":
                sim = get_simulation()
                status = sim.get_status()
                status["use_llm"] = config.USE_LLM
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "data": status,
                }, ensure_ascii=False))

            elif msg.get("action") == "auto_run":
                sim = get_simulation()
                steps = msg.get("steps", 5)
                for i in range(steps):
                    state = await run_step_with_progress(sim)
                    step_data = _build_step_data(state, sim)
                    await websocket.send_text(json.dumps(
                        {"type": "step_update", "data": step_data},
                        ensure_ascii=False,
                    ))
                    await asyncio.sleep(1)

    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


def start_server():
    """启动 Web 服务器"""
    import uvicorn
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)
