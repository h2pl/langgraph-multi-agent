"""模拟时钟 - 管理小镇的时间流逝"""

from dataclasses import dataclass


@dataclass
class SimClock:
    """模拟时钟，每个 step 代表 1 小时"""
    day: int = 1
    hour: int = 7  # 从早上 7 点开始

    def tick(self) -> None:
        """推进 1 小时"""
        self.hour += 1
        if self.hour >= 23:  # 晚上 11 点结束一天
            self.hour = 7
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
        return self.hour >= 22

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "hour": self.hour,
            "time_str": self.time_str,
            "period": self.period,
        }
