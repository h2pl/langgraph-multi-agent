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


def _format_event(time_str: str, content: str) -> str:
    """为事件统一加模拟时间戳：[hh:00] 原内容"""
    return f"[{time_str.split(' ', 1)[-1]}] {content}"

logger = logging.getLogger(__name__)

app = FastAPI(title="桃源镇 - AI 小镇模拟")

# 静态文件
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 全局模拟实例
simulation: TownSimulation | None = None
connected_clients: list[WebSocket] = []

# 通用取消信号（用于 autoRun / runDay / 单步中断）
_cancel_event = asyncio.Event()


class SimulationCancelled(Exception):
    """模拟被用户取消"""
    pass


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


async def broadcast_ui_state(
    *,
    buttons_disabled: bool | None = None,
    day_running: bool | None = None,
    auto_running: bool | None = None,
    step_running: bool | None = None,
    elapsed: str | None = None,
):
    """推送 UI 状态（按钮禁用、运行状态、计时），前端只负责渲染"""
    payload: dict = {"type": "ui_state", "data": {}}
    if buttons_disabled is not None:
        payload["data"]["buttons_disabled"] = buttons_disabled
    if day_running is not None:
        payload["data"]["day_running"] = day_running
    if auto_running is not None:
        payload["data"]["auto_running"] = auto_running
    if step_running is not None:
        payload["data"]["step_running"] = step_running
    if elapsed is not None:
        payload["data"]["elapsed"] = elapsed
    message = json.dumps(payload, ensure_ascii=False)
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            pass


PHASE_LABELS_GLOBAL = {"perceive": "感知", "plan": "计划", "act": "行动",
                      "interact": "处理同一地点的居民交互",
                      "advance": "把时间推进到下一小时",
                      "reflect": "反思当天的经历",
                      "dispatch_finishing": "处理同一地点的居民交互"}
LLM_PHASES_GLOBAL = {"plan", "reflect"}


async def broadcast_micro_step(agent_name: str | None, phase: str, events: list,
                               agent_state: dict | None, location_states: dict):
    """向所有客户端推送单个居民单阶段完成后的实时事件"""
    payload = {
        "agent_name": agent_name,
        "phase": phase,
        "phase_label": PHASE_LABELS_GLOBAL.get(phase, phase),
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


async def broadcast_phase_start(agent_name: str | None, phase: str | None):
    """向所有客户端推送阶段开始信息，用于显示'正在XXX'"""
    phase_label = PHASE_LABELS_GLOBAL.get(phase or "", phase or "")
    is_llm = phase in LLM_PHASES_GLOBAL if phase else False
    payload = {
        "agent_name": agent_name,
        "phase": phase,
        "phase_label": phase_label,
        "is_llm": is_llm,
    }
    message = json.dumps({"type": "phase_start", "data": payload}, ensure_ascii=False)
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


async def _run_one_hour(sim: TownSimulation) -> dict:
    """原子：循环 micro_step 直到完成一小时（dispatch/interact/advance 全部完成）

    所有实时推送在 micro_step 内部已经通过 broadcast_* 完成。
    返回最后一步的完整状态（包含 finished_hour=True）。
    """
    while True:
        # 推送"XX正在XX"（下一步的 agent/phase）
        # - dispatch 阶段具体 agent 的 perceive/plan/act → 推送该 agent
        # - 即将进入 interact 阶段 → 推送"正在处理同一地点的居民交互"
        # - reflect 阶段 → 推送"反思当天的经历"
        # - advance / dispatch_finishing 是瞬间完成的协调步骤，不打扰状态栏
        if _cancel_event.is_set():
            raise SimulationCancelled("用户取消")
        info = sim.get_micro_info()
        phase_name = info.get("phase_name")
        agent_name = info.get("agent_name")
        if phase_name in {"perceive", "plan", "act"} and agent_name:
            await broadcast_phase_start(agent_name, phase_name)
        elif phase_name in {"dispatch_finishing", "interact"}:
            await broadcast_phase_start(None, "interact")
        elif phase_name == "reflect":
            await broadcast_phase_start(None, "reflect")

        result = await asyncio.get_event_loop().run_in_executor(None, sim.micro_step)

        # 每个微步完成后立即检查暂停，避免长 hour 中暂停响应慢
        if _cancel_event.is_set():
            raise SimulationCancelled("用户取消")

        # 实时推送微步结果（保持与原来 do_micro_step 一致）
        await broadcast_micro_step(
            result.get("agent_name"),
            result["phase"],
            result.get("events", []),
            result.get("agent_state"),
            result.get("location_states", {}),
        )

        # 推送对话
        for conv in result.get("conversations", []):
            event_text = _format_event(
                sim.clock.time_str,
                f"[对话] {' & '.join(conv['participants'])} 在{conv['location']}聊天",
            )
            await broadcast_conversation(event_text, conv)

        # interact 阶段刚完成 → 状态栏回到"就绪"，避免停留在误导性的 agent 文字上
        if result.get("phase") == "interact":
            await broadcast_progress("done", "居民交互已处理")

        # 完成一小时时推送全局状态刷新
        if result.get("finished_hour"):
            await broadcast_progress("done", "本小时完成")
            step_data = {
                "time": sim.clock.time_str,
                "sim_time": sim.clock.time_str,
                "sim_period": sim.clock.period,
                "sim_day": sim.clock.day,
                "events": result.get("events", []),
                "conversations": result.get("conversations", []),
                "agent_states": result.get("agent_states")
                or {n: r.to_dict() for n, r in sim.residents.items()},
                "location_states": result.get("location_states", {}),
            }
            msg = json.dumps({"type": "step_update", "data": step_data}, ensure_ascii=False)
            for client in connected_clients:
                try:
                    await client.send_text(msg)
                except Exception:
                    pass
            return result


@app.post("/api/run_hour")
async def run_hour():
    """运行下一个小时（循环 micro_step 直到 finished_hour）"""
    sim = get_simulation()
    _cancel_event.clear()
    await broadcast_ui_state(buttons_disabled=True, step_running=True)
    try:
        result = await _run_one_hour(sim)
    except SimulationCancelled:
        await broadcast_ui_state(buttons_disabled=False, step_running=False)
        await broadcast_progress("stopped", "已停止")
        return {"status": "stopped", "message": "已停止"}
    except Exception as e:
        logger.exception("run_hour failed")
        await broadcast_ui_state(buttons_disabled=False, step_running=False)
        return {"status": "error", "message": str(e)}

    await broadcast_ui_state(buttons_disabled=False, step_running=False)
    return {
        "status": "ok",
        "clock": {
            "day": sim.clock.day,
            "time_str": sim.clock.time_str,
            "period": sim.clock.period,
        },
    }


@app.post("/api/run_day")
async def run_day():
    """运行一整天（循环 run_hour 直到天数变化）"""
    import time as _time
    sim = get_simulation()
    _cancel_event.clear()
    t0 = _time.monotonic()

    await broadcast_ui_state(buttons_disabled=True, day_running=True)

    start_day = sim.clock.day
    step = 0

    while sim.clock.day == start_day:
        if _cancel_event.is_set():
            elapsed = f"{_time.monotonic() - t0:.1f}s"
            await broadcast_ui_state(buttons_disabled=False, day_running=False)
            await broadcast_progress(
                "stopped", f"已停止（第 {step} 步，耗时 {elapsed}）"
            )
            return {"status": "stopped", "step": step, "elapsed": elapsed}

        step += 1
        elapsed = f"{_time.monotonic() - t0:.1f}s"
        await broadcast_progress(
            "day_running",
            f"第 {step} 步 · {sim.clock.time_str} · 已用时 {elapsed}",
            {"elapsed": elapsed, "step": step, "sim_time": sim.clock.time_str},
        )
        await broadcast_ui_state(elapsed=elapsed)

        try:
            await _run_one_hour(sim)
        except SimulationCancelled:
            elapsed = f"{_time.monotonic() - t0:.1f}s"
            await broadcast_ui_state(buttons_disabled=False, day_running=False)
            await broadcast_progress(
                "stopped", f"已停止（第 {step} 步，耗时 {elapsed}）"
            )
            return {"status": "stopped", "step": step, "elapsed": elapsed}
        except Exception as e:
            logger.exception("run_day step %d failed", step)
            await broadcast_progress("error", f"第 {step} 步执行失败: {e}")
            break

        await asyncio.sleep(1.0)

    elapsed = f"{_time.monotonic() - t0:.1f}s"
    await broadcast_progress("done", f"一天模拟完成（耗时 {elapsed}）")
    await broadcast_ui_state(buttons_disabled=False, day_running=False)
    return {"status": "ok", "step": step, "elapsed": elapsed}


@app.post("/api/micro_step")
async def do_micro_step():
    """执行一个微步（当前居民的下一个 perceive/plan/act 阶段）"""
    sim = get_simulation()

    # 推送"XX正在XX"；dispatch 阶段具体 agent 的动作、或即将进入 interact/reflect
    info = sim.get_micro_info()
    phase_name = info.get("phase_name")
    agent_name = info.get("agent_name")
    if phase_name in {"perceive", "plan", "act"} and agent_name:
        await broadcast_phase_start(agent_name, phase_name)
    elif phase_name in {"dispatch_finishing", "interact"}:
        await broadcast_phase_start(None, "interact")
    elif phase_name == "reflect":
        await broadcast_phase_start(None, "reflect")

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

    # interact 阶段刚完成 → 状态栏回到"就绪"
    if result.get("phase") == "interact":
        await broadcast_progress("done", "居民交互已处理")

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

    return {
        "status": "ok",
        "result": result,
        "clock": {
            "day": sim.clock.day,
            "time_str": sim.clock.time_str,
            "period": sim.clock.period,
        },
    }


@app.get("/api/micro_info")
async def micro_info():
    """获取当前微步进度"""
    sim = get_simulation()
    return sim.get_micro_info()


@app.post("/api/pause_day")
async def pause_day():
    """停止正在运行的模拟（通用）"""
    _cancel_event.set()
    return {"status": "ok", "message": "停止信号已发送"}


@app.post("/api/clear_pause")
async def clear_pause():
    """清除暂停标志（开始新一轮运行前调用）"""
    _cancel_event.clear()
    return {"status": "ok"}


@app.post("/api/set_time")
async def set_time(payload: dict):
    """将模拟时钟调到指定整点（day >=1, hour: 7~22）"""
    day = payload.get("day")
    hour = payload.get("hour")
    if day is None or hour is None:
        return {"status": "error", "message": "缺少 day 或 hour 参数"}
    try:
        sim = get_simulation()
        old_time = sim.clock.time_str
        sim.set_time(day, hour)
        new_time = sim.clock.time_str
        clock_dict = sim.clock.to_dict()
        await broadcast_progress(
            "set_time",
            f"时间已调至第{day}天 {hour:02d}:00",
            {"clock": clock_dict},
        )
        cleared = None
        if new_time != old_time:
            cleared = f"已清除 {old_time} ~ {new_time} 期间的记忆"
            await broadcast_progress("memory_cleared", cleared)
        return {"status": "ok", "clock": clock_dict, "cleared": cleared, "old_time": old_time}
    except ValueError as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/reset")
async def reset():
    """重置模拟到第一天"""
    global simulation
    old_time = (get_simulation() if simulation else None)
    if old_time:
        old_time = old_time.clock.time_str
    else:
        old_time = ""
    _cancel_event.set()
    await asyncio.sleep(0.1)
    _cancel_event.clear()
    await broadcast_progress("reset", "正在重置模拟...")
    simulation = TownSimulation()
    new_time = simulation.clock.time_str
    await broadcast_ui_state(buttons_disabled=False, step_running=False,
                             auto_running=False, day_running=False)
    await broadcast_progress("done", "重置完成")
    cleared = None
    if old_time and old_time != new_time:
        cleared = f"已清除 {old_time} ~ {new_time} 期间的记忆"
    return {"status": "ok", "clock": simulation.clock.to_dict(), "cleared": cleared, "old_time": old_time}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 实时推送"""
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("action") == "status":
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
                delay = msg.get("delay", 1)
                _cancel_event.clear()
                await broadcast_ui_state(buttons_disabled=True, auto_running=True)
                cancelled = False
                for i in range(steps):
                    if _cancel_event.is_set():
                        cancelled = True
                        break
                    try:
                        await _run_one_hour(sim)
                    except SimulationCancelled:
                        cancelled = True
                        break
                    except Exception:
                        logger.exception("auto_run step failed")
                        break
                    if i < steps - 1:
                        await asyncio.sleep(delay)
                if cancelled:
                    await broadcast_progress("stopped", "自动运行已停止")
                await broadcast_ui_state(buttons_disabled=False, auto_running=False)

    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


def start_server():
    """启动 Web 服务器"""
    import uvicorn
    uvicorn.run("web.app:app", host=config.WEB_HOST, port=config.WEB_PORT, reload=True)
