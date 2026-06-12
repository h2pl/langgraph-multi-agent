"""Agent 交互逻辑 - 对话和行为

支持两种模式（由 config.USE_LLM 控制）:
  - LLM 模式: 调用大模型生成丰富的行为和对话
  - 随机模式: 系统随机模拟，不消耗 API 配额
"""

from __future__ import annotations

import random
import logging

from simulation.llm_utils import llm_call_sync
from agents.resident import Resident
from config import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  随机模拟用的模板数据
# ═══════════════════════════════════════════════════════════════

# 按时间段划分的活动模板: {period: [(activity, emotion), ...]}
_PERIOD_ACTIVITIES = {
    "清晨": [
        ("晨练", "精神"),
        ("准备早餐", "平静"),
        ("打扫卫生", "平静"),
        ("早起散步", "愉快"),
    ],
    "上午": [
        ("工作", "专注"),
        ("处理事务", "平静"),
        ("整理东西", "平静"),
        ("看书", "专注"),
        ("和邻居聊天", "愉快"),
    ],
    "中午": [
        ("吃午饭", "满足"),
        ("午休", "平静"),
        ("闲逛", "悠闲"),
        ("买东西", "平静"),
    ],
    "下午": [
        ("继续工作", "专注"),
        ("喝茶休息", "悠闲"),
        ("散步", "愉快"),
        ("拜访朋友", "愉快"),
        ("处理杂事", "平静"),
    ],
    "傍晚": [
        ("准备晚饭", "平静"),
        ("在广场散步", "悠闲"),
        ("和人聊天", "愉快"),
        ("收拾东西", "平静"),
    ],
    "晚上": [
        ("看电视", "悠闲"),
        ("读书", "平静"),
        ("和家人聊天", "温馨"),
        ("准备休息", "平静"),
    ],
}

# 通用活动（任何时间段都可以）
_GENERIC_ACTIVITIES = [
    ("四处闲逛", "悠闲"),
    ("发呆", "平静"),
    ("想事情", "若有所思"),
]

# 随机对话模板
_CONVERSATION_TEMPLATES = [
    [
        "{a1}: 嗨，{a2}，你今天怎么样？",
        "{a2}: 还不错啊，你呢？",
        "{a1}: 挺好的，就是有点忙。",
        "{a2}: 忙点好，说明生活充实嘛。",
    ],
    [
        "{a1}: {a2}，好久没见你了！",
        "{a2}: 是啊，最近都忙什么呢？",
        "{a1}: 还是老样子，{act1}。你呢？",
        "{a2}: 我也差不多，{act2}。",
        "{a1}: 有空一起坐坐啊。",
    ],
    [
        "{a1}: 今天天气真不错。",
        "{a2}: 是啊，适合出来走走。",
        "{a1}: 你这是要去哪啊？",
        "{a2}: 随便逛逛，透透气。",
    ],
    [
        "{a1}: {a2}，吃了没？",
        "{a2}: 还没呢，正想着去哪吃。",
        "{a1}: 走，一起去老王面馆吧。",
        "{a2}: 好主意，走！",
    ],
    [
        "{a1}: 最近镇上有什么新鲜事吗？",
        "{a2}: 没什么大事，就是日子过得挺平静的。",
        "{a1}: 平静就好，平平安安的。",
        "{a2}: 说得对，知足常乐。",
    ],
]

# 反思模板
_REFLECTION_TEMPLATES = [
    "今天过得还算充实，{name}觉得这样的日子挺好的。",
    "{name}想着，应该多和邻居们走动走动。",
    "忙了一天，{name}觉得有些累，但心里很踏实。",
    "{name}觉得小镇的生活虽然平淡，但自有它的温暖。",
    "回想今天的事，{name}嘴角不自觉地上扬了。",
    "{name}决定明天要早点起来，好好安排一下。",
    "今天和大家聊了聊，{name}觉得心情好了不少。",
    "{name}想，生活不就是这样一天天过的嘛。",
]


# ═══════════════════════════════════════════════════════════════
#  随机模拟实现
# ═══════════════════════════════════════════════════════════════

def _random_plan(agent: Resident, time_str: str, period: str, location_options: list[str]) -> dict:
    """系统随机生成行动计划（不调用 LLM）"""
    # 根据时间段选活动
    activities = _PERIOD_ACTIVITIES.get(period, _GENERIC_ACTIVITIES)
    activity, emotion = random.choice(activities)

    # 根据职业和时间段决定大概率去哪
    workplace = getattr(agent, "workplace", None)
    home = getattr(agent, "home", agent.current_location)

    if period in ("清晨", "晚上"):
        # 早晚大概率在家
        location = home if home in location_options else random.choice(location_options)
    elif period in ("上午", "下午") and workplace and workplace in location_options:
        # 工作时间大概率在工作地点
        location = workplace if random.random() < 0.7 else random.choice(location_options)
    else:
        # 其他时间随机
        # 偏好公共场所和商业区
        public_locs = [l for l in location_options if "家" not in l]
        if public_locs and random.random() < 0.6:
            location = random.choice(public_locs)
        else:
            location = random.choice(location_options)

    return {
        "location": location,
        "activity": activity,
        "emotion": emotion,
    }


def _random_conversation(agent1: Resident, agent2: Resident, context: str) -> str:
    """系统随机生成对话（不调用 LLM）"""
    template = random.choice(_CONVERSATION_TEMPLATES)
    lines = []
    for line in template:
        lines.append(line.format(
            a1=agent1.name,
            a2=agent2.name,
            act1=agent1.current_activity or "忙活",
            act2=agent2.current_activity or "忙活",
        ))
    return "\n".join(lines)


def _random_reflection(agent: Resident, time_str: str) -> str:
    """系统随机生成反思（不调用 LLM）"""
    return random.choice(_REFLECTION_TEMPLATES).format(name=agent.name)


# ═══════════════════════════════════════════════════════════════
#  公共接口（自动根据 config.USE_LLM 选择模式）
# ═══════════════════════════════════════════════════════════════

def generate_conversation(agent1: Resident, agent2: Resident, context: str) -> str:
    """生成两个居民之间的对话"""
    if not config.USE_LLM:
        return _random_conversation(agent1, agent2, context)

    # RAG: 双方检索与对方的过往互动，融入当前行为上下文
    a1_about_a2 = agent1.memory.retrieve(
        query=f"和{agent2.name}的交往对话 {agent2.name}正在{agent2.current_activity}",
        top_k=5,
    )
    a2_about_a1 = agent2.memory.retrieve(
        query=f"和{agent1.name}的交往对话 {agent1.name}正在{agent1.current_activity}",
        top_k=5,
    )
    a1_memory_text = agent1.memory.format_memories(a1_about_a2)
    a2_memory_text = agent2.memory.format_memories(a2_about_a1)

    system_prompt = (
        "你是一个小镇生活模拟器。请根据两个角色的性格、关系和过往记忆，生成一段自然的对话。\n"
        "对话应该简短（3-5轮），符合角色性格，有生活气息，且体现过往互动的连续性。\n"
        "只输出对话内容，格式为：\n"
        "角色名: 对话内容\n"
    )

    rel_1to2 = agent1.relationships.get(agent2.name, "不太熟悉")
    rel_2to1 = agent2.relationships.get(agent1.name, "不太熟悉")

    user_prompt = (
        f"场景: {context}\n\n"
        f"角色1: {agent1.name}（{agent1.occupation}，性格: {agent1.personality}）\n"
        f"  当前状态: {agent1.status_summary}\n"
        f"  对{agent2.name}的看法: {rel_1to2}\n"
        f"  关于{agent2.name}的记忆:\n{a1_memory_text}\n\n"
        f"角色2: {agent2.name}（{agent2.occupation}，性格: {agent2.personality}）\n"
        f"  当前状态: {agent2.status_summary}\n"
        f"  对{agent1.name}的看法: {rel_2to1}\n"
        f"  关于{agent1.name}的记忆:\n{a2_memory_text}\n\n"
        f"请生成他们的对话:"
    )

    return llm_call_sync(system_prompt, user_prompt)


def rate_importance(observation: str, agent: Resident) -> float:
    """评估一条观察的重要性（1-10）— 使用规则而非 LLM，节省 API 配额"""
    # 关键词匹配打分，避免浪费 LLM 调用
    high_keywords = ["生病", "受伤", "吵架", "哭", "紧急", "出事", "危险", "失火"]
    mid_keywords = ["聊天", "对话", "帮忙", "拜访", "约", "邀请", "一起"]
    low_keywords = ["闲逛", "散步", "路过", "看到", "发呆"]

    for kw in high_keywords:
        if kw in observation:
            return 8.0
    for kw in mid_keywords:
        if kw in observation:
            return 6.0
    for kw in low_keywords:
        if kw in observation:
            return 3.0
    return 5.0


def generate_plan(agent: Resident, time_str: str, period: str,
                  location_options: list[str], observation: str = "") -> dict:
    """生成居民的行动计划"""
    if not config.USE_LLM:
        return _random_plan(agent, time_str, period, location_options)

    # RAG: 基于当前上下文动态构造检索 query（斯坦福论文：query 即当前处境描述）
    query_parts = []
    if observation:
        query_parts.append(observation[:80])  # 观察是最相关的上下文
    query_parts.append(f"{period}时间的安排和去处")
    if agent.current_activity and agent.current_activity != "休息":
        query_parts.append(f"正在{agent.current_activity}")
    query = " ".join(query_parts)

    relevant_memories = agent.memory.retrieve(query=query, top_k=8)
    memory_text = agent.memory.format_memories(relevant_memories)

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
    if not config.USE_LLM:
        return _random_reflection(agent, time_str)

    # RAG: 用最近观察构造上下文驱动的检索 query（斯坦福论文：反思基于具体经历）
    recent_obs = agent.memory.get_recent_observations(3)
    if recent_obs:
        obs_keywords = "；".join(m.content[:40] for m in recent_obs)
        query = f"关于这些经历的感受：{obs_keywords}"
    else:
        query = "最近重要的事和印象深刻的经历"

    relevant_memories = agent.memory.retrieve(query=query, top_k=10)
    memory_text = agent.memory.format_memories(relevant_memories)

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
