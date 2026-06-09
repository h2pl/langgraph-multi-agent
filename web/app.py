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


def get_simulation() -> TownSimulation:
    global simulation
    if simulation is None:
        simulation = TownSimulation()
    return simulation


async def broadcast_progress(phase: str, detail: str = ""):
    """向所有 WebSocket 客户端广播进度消息"""
    message = json.dumps({
        "type": "progress",
        "data": {"phase": phase, "detail": detail}
    }, ensure_ascii=False)

    dead_clients = []
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            dead_clients.append(client)
    for c in dead_clients:
        if c in connected_clients:
            connected_clients.remove(c)


async def run_step_with_progress(sim: TownSimulation) -> dict:
    """运行单步并通过 WebSocket 发送进度"""
    state = sim.get_initial_state()

    # 阶段 1: 调度 Agents
    mode_label = "LLM 生成中" if config.USE_LLM else "随机模拟中"
    await broadcast_progress("dispatch", f"正在调度居民行为 ({mode_label})...")
    state = await asyncio.get_event_loop().run_in_executor(
        None, lambda: {**state, **sim._dispatch_agents_node(state)}
    )

    # 阶段 2: 交互
    await broadcast_progress("interact", "正在处理居民交互...")
    state = await asyncio.get_event_loop().run_in_executor(
        None, lambda: {**state, **sim._interact_node(state)}
    )

    # 阶段 3: 推进时间
    await broadcast_progress("advance", "正在推进时间...")
    state = await asyncio.get_event_loop().run_in_executor(
        None, lambda: {**state, **sim._advance_time_node(state)}
    )

    await broadcast_progress("done", "步骤完成")
    return state


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
    state = await run_step_with_progress(sim)

    # 通知所有 WebSocket 客户端
    message = json.dumps({
        "type": "step_update",
        "data": {
            "time": state["time_str"],
            "events": state["events"][-10:],
            "conversations": state.get("conversations", []),
            "agent_states": state["agent_states"],
            "location_states": state["location_states"],
        }
    }, ensure_ascii=False)

    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            pass

    return {"status": "ok", "state": state}


@app.post("/api/run_day")
async def run_day():
    """运行一整天"""
    sim = get_simulation()
    await broadcast_progress("day_start", "开始运行一整天的模拟...")

    state = await asyncio.get_event_loop().run_in_executor(
        None, sim.run_day
    )

    await broadcast_progress("done", "一天模拟完成")
    return {
        "status": "ok",
        "day": state["day"],
        "events": state["day_log"],
        "conversations": state["conversations"],
    }


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
                await websocket.send_text(json.dumps({
                    "type": "step_update",
                    "data": {
                        "time": state["time_str"],
                        "events": state["events"][-10:],
                        "conversations": state.get("conversations", []),
                        "agent_states": state["agent_states"],
                        "location_states": state["location_states"],
                    }
                }, ensure_ascii=False))

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
                    await websocket.send_text(json.dumps({
                        "type": "step_update",
                        "data": {
                            "time": state["time_str"],
                            "events": state["events"][-10:],
                            "conversations": state.get("conversations", []),
                            "agent_states": state["agent_states"],
                            "location_states": state["location_states"],
                        }
                    }, ensure_ascii=False))
                    await asyncio.sleep(1)

    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


def start_server():
    """启动 Web 服务器"""
    import uvicorn
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)
