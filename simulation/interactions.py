"""Agent 交互逻辑 - 对话和行为"""

from __future__ import annotations

from simulation.llm_utils import llm_call_sync
from agents.resident import Resident


def generate_conversation(agent1: Resident, agent2: Resident, context: str) -> str:
    """生成两个居民之间的对话"""
    system_prompt = (
        "你是一个小镇生活模拟器。请根据两个角色的性格和关系，生成一段自然的对话。\n"
        "对话应该简短（3-5轮），符合角色性格，有生活气息。\n"
        "只输出对话内容，格式为：\n"
        "角色名: 对话内容\n"
    )

    rel_1to2 = agent1.relationships.get(agent2.name, "不太熟悉")
    rel_2to1 = agent2.relationships.get(agent1.name, "不太熟悉")

    user_prompt = (
        f"场景: {context}\n\n"
        f"角色1: {agent1.name}（{agent1.occupation}，性格: {agent1.personality}）\n"
        f"  当前状态: {agent1.status_summary}\n"
        f"  对{agent2.name}的看法: {rel_1to2}\n\n"
        f"角色2: {agent2.name}（{agent2.occupation}，性格: {agent2.personality}）\n"
        f"  当前状态: {agent2.status_summary}\n"
        f"  对{agent1.name}的看法: {rel_2to1}\n\n"
        f"请生成他们的对话:"
    )

    return llm_call_sync(system_prompt, user_prompt)


def rate_importance(observation: str, agent: Resident) -> float:
    """评估一条观察的重要性（1-10）"""
    system_prompt = (
        "请给以下事件对这个人的重要性打分（1-10分）。\n"
        "1分表示完全不重要（如：看到一片树叶落下）\n"
        "5分表示一般重要（如：和邻居打了个招呼）\n"
        "10分表示非常重要（如：得知亲人生病）\n"
        "只输出一个数字，不要其他内容。"
    )
    user_prompt = (
        f"人物: {agent.name}（{agent.occupation}）\n"
        f"事件: {observation}"
    )

    try:
        result = llm_call_sync(system_prompt, user_prompt)
        score = float(result.strip())
        return max(1.0, min(10.0, score))
    except (ValueError, TypeError):
        return 5.0


def generate_plan(agent: Resident, time_str: str, period: str, location_options: list[str]) -> dict:
    """生成居民的行动计划"""
    recent_memories = agent.memory.get_recent_observations(5)
    memory_text = agent.memory.format_memories(recent_memories)

    system_prompt = (
        "你是一个小镇生活模拟器。请根据角色信息，决定这个角色接下来要做什么。\n"
        "请输出严格的JSON格式（不要 markdown 代码块），包含以下字段：\n"
        '{"location": "目标地点", "activity": "要做的事", "emotion": "当前心情"}\n'
        "地点必须从可选地点中选择。活动要具体、自然、符合角色特征。"
    )

    user_prompt = (
        f"当前时间: {time_str}（{period}）\n\n"
        f"角色信息:\n{agent.profile_summary}\n\n"
        f"当前状态: {agent.status_summary}\n\n"
        f"最近的记忆:\n{memory_text}\n\n"
        f"可选地点: {', '.join(location_options)}\n\n"
        f"请决定{agent.name}接下来要做什么:"
    )

    try:
        import json
        result = llm_call_sync(system_prompt, user_prompt)
        # 尝试清理 markdown 代码块
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[-1]
            result = result.rsplit("```", 1)[0]
        return json.loads(result.strip())
    except (json.JSONDecodeError, Exception):
        return {
            "location": agent.current_location,
            "activity": "四处闲逛",
            "emotion": "平静",
        }


def generate_reflection(agent: Resident, time_str: str) -> str:
    """生成反思"""
    recent = agent.memory.get_recent_observations(10)
    memory_text = agent.memory.format_memories(recent)

    system_prompt = (
        "你是一个小镇居民的内心独白生成器。\n"
        "请根据这个角色最近的经历，生成一段简短的反思或感悟（1-2句话）。\n"
        "反思应该体现角色的性格特征，并可能影响未来的行为。\n"
        "只输出反思内容，不要其他格式。"
    )

    user_prompt = (
        f"角色: {agent.name}（{agent.personality}）\n\n"
        f"最近的经历:\n{memory_text}\n\n"
        f"请生成{agent.name}的内心反思:"
    )

    return llm_call_sync(system_prompt, user_prompt)
