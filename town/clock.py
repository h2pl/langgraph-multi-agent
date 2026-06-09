"""模拟时钟 - 管理小镇的时间流逝"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimClock:
    """模拟时钟，每个 step 代表 1 小时

    设计说明：
    - 时间范围 7:00 ~ 22:00（即 hour 7..22，共 16 个时段）
    - hour == 22 时标记为一天结束（is_day_end == True），
      由 Supervisor 触发 reflect；reflect 之后通过 `advance_to_next_day()` 进入下一天。
    - 这样可以确保 22:00 这个时段本身被处理，而不是被直接跳过。
    """
    day: int = 1
    hour: int = 7  # 从早上 7 点开始
    DAY_START_HOUR: int = 7
    DAY_END_HOUR: int = 22

    def tick(self) -> None:
        """推进 1 小时；当到达 22:00 时停下，等待 reflect 完成后再翻页。"""
        if self.hour < self.DAY_END_HOUR:
            self.hour += 1

    def advance_to_next_day(self) -> None:
        """一天真正结束后的翻页操作"""
        self.hour = self.DAY_START_HOUR
        self.day += 1

    @property
    def time_str(self) -> str:
        return f"第{self.day}天 {self.hour:02d}:00"

    @property
    def period(self) -> str:
        """返回当前时间段"""
        if 7 <= self.hour < 9:
            return "清晨"
        elif 9 <= self.hour < 12:
            return "上午"
        elif 12 <= self.hour < 14:
            return "中午"
        elif 14 <= self.hour < 18:
            return "下午"
        elif 18 <= self.hour < 20:
            return "傍晚"
        else:
            return "夜晚"

    @property
    def is_day_end(self) -> bool:
        return self.hour >= self.DAY_END_HOUR

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "hour": self.hour,
            "time_str": self.time_str,
            "period": self.period,
        }
