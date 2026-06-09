"""桃源镇 - AI 小镇模拟 入口文件

用法:
    python main.py web      启动 Web 可视化界面
    python main.py console  在终端运行一天的模拟
    python main.py step     在终端单步运行
"""

import sys
import json
import logging
logger = logging.getLogger(__name__)

from config import config


def run_console():
    """终端模式：运行一天的模拟"""
    from simulation.engine import TownSimulation

    print("=" * 60)
    print("--- Step mode started ---")
    print("=" * 60)

    sim = TownSimulation()
    print(f"\n📍 小镇: {sim.town.name}")
    print(f"👥 居民: {', '.join(sim.residents.keys())}")
    print(f"🕐 时间: {sim.clock.time_str}\n")
    print("-" * 60)

    state = sim.run_day()

    print("\n📋 今日事件:")
    print("-" * 60)
    for event in state["day_log"]:
        print(f"  {event}")

    if state["conversations"]:
        print("\n💬 今日对话:")
        print("-" * 60)
        for conv in state["conversations"]:
            print(f"\n  [{conv['time']}] {' & '.join(conv['participants'])} @ {conv['location']}")
            print(f"  {'─' * 40}")
            for line in conv["content"].split("\n"):
                print(f"    {line}")

    print("\n🌙 一天结束")
    print("=" * 60)

    # 展示居民状态
    print("\n👥 居民最终状态:")
    for name, info in state["agent_states"].items():
        memory_info = info.get("memory", {})
        print(f"  {name}: {info['current_location']} | "
              f"记忆: {memory_info.get('total_memories', 0)}条 "
              f"(观察{memory_info.get('observations', 0)} + "
              f"反思{memory_info.get('reflections', 0)})")


def run_step_mode():
    """终端单步模式"""
    from simulation.engine import TownSimulation

    logger.info("--- Step mode started ---")
    logger.info("Press Enter to execute next step, type 'q' to quit")
    sim = TownSimulation()
    while True:
        cmd = input(f"[{sim.clock.time_str}] >>> ").strip()
        if cmd:
            state = sim.run_step()
            logger.info("Executed step at %s", sim.clock.time_str)
            logger.debug("Events: %s", state["events"][-5:])
        if cmd.lower() == "q":
            break

        state = sim.run_step()

        for event in state["events"]:
            print(f"  {event}")

        if state.get("conversations"):
            for conv in state["conversations"]:
                print(f"\n  💬 {' & '.join(conv['participants'])}:")
                for line in conv["content"].split("\n"):
                    print(f"    {line}")

        print()


def run_web():
    """Web 模式"""
    from web.app import start_server

    print("🏘️  桃源镇 · Web 可视化模式")
    print(f"🌐 打开浏览器访问: http://{config.WEB_HOST}:{config.WEB_PORT}")
    print("按 Ctrl+C 停止\n")
    start_server()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("请选择运行模式: web / console / step")
        return

    mode = sys.argv[1].lower()

    if not config.OPENAI_API_KEY:
        print("⚠️  请先设置 OPENAI_API_KEY!")
        print("   1. 复制 .env.example 为 .env")
        print("   2. 填入你的 API Key")
        return

    if mode == "web":
        run_web()
    elif mode == "console":
        run_console()
    elif mode == "step":
        run_step_mode()
    else:
        print(f"未知模式: {mode}")
        print("可用模式: web / console / step")


if __name__ == "__main__":
    main()
