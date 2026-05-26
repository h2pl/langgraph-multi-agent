"""Web 可视化 - FastAPI + WebSocket 实时展示小镇模拟"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from simulation.engine import TownSimulation
from config import config

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


@app.get("/", response_class=HTMLResponse)
async def index():
    """主页"""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status():
    """获取当前状态"""
    sim = get_simulation()
    return sim.get_status()


@app.post("/api/step")
async def run_step():
    """执行一步模拟"""
    sim = get_simulation()
    state = sim.run_step()

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
    state = sim.run_day()
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
    simulation = TownSimulation()
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
                state = sim.run_step()
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
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "data": sim.get_status(),
                }, ensure_ascii=False))

            elif msg.get("action") == "auto_run":
                sim = get_simulation()
                steps = msg.get("steps", 5)
                for _ in range(steps):
                    state = sim.run_step()
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
        connected_clients.remove(websocket)


def start_server():
    """启动 Web 服务器"""
    import uvicorn
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)
