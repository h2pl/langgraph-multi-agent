"""临时烟雾测试脚本"""
import os
os.environ["USE_LLM"] = "false"

from config import config
config.USE_LLM = False

from simulation.engine import TownSimulation

sim = TownSimulation()
print("=== INIT ===")
print(f"start: {sim.clock.time_str} | {sim.clock.period} | day_end={sim.clock.is_day_end}")
print(f"residents: {list(sim.residents.keys())}")
print()

for i in range(3):
    s = sim.run_step()
    print(f"--- step {i+1} -> now {sim.clock.time_str} ({sim.clock.period}) day_end={sim.clock.is_day_end} ---")
    for e in s["events"]:
        print(f"  E: {e}")
    for c in s["conversations"]:
        head = c["content"][:60].replace("\n", " | ")
        print(f"  C[{c['time']}] {c['participants']} @ {c['location']}: {head}...")
    print()

# 一直跑到 22 点
print("=== run until day end ===")
for i in range(20):
    if sim.clock.is_day_end:
        break
    sim.run_step()

print(f"now: {sim.clock.time_str} | day_end={sim.clock.is_day_end}")
if sim.clock.is_day_end:
    print("--- end_day ---")
    s = sim.end_day()
    for e in s["events"]:
        print(f"  E: {e}")
    print(f"after reflect: {sim.clock.time_str} (day {sim.clock.day})")
