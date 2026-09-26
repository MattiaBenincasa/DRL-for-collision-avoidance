#!/usr/bin/env python3

import os
import sys
import time

repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

import rospy
import numpy as np
from tensorflow.keras.models import load_model

from platform_layer.scripts.CommandManager import CommandManager
from platform_layer.scripts.StatusReporter import StatusReporter
from platform_layer.scripts.LaserSensor import LaserSensor
from simulation_layer.scripts.SimulationManager import SimulationManager
from control_layer.scripts.DDQNAgent import DDQNAgent

rospy.init_node('storm_test_node')

state_size = 50
action_size = 11

model_path = os.path.join(repo_src, 'models', 'm3.keras')
simulation_manager = SimulationManager('map_3')
laser_sensor = LaserSensor()
status_reporter = StatusReporter(laser_sensor=laser_sensor)
command_manager = CommandManager(status_reporter)

def _noop_reward(*args, **kwargs):
    return 0.0, False

agent = DDQNAgent(state_size, action_size, reward_func=_noop_reward)
agent.epsilon = 0.0

if os.path.exists(model_path):
    agent.eval_model = load_model(model_path)
    print(f"Model loaded: {model_path}")
else:
    print(f"Model not found: {model_path}")
    sys.exit(1)

simulation_manager.respawn(parallel_spawn=True)
time.sleep(0.6)

state = np.reshape(status_reporter.get_state(), [1, state_size])

duration = 300
rate = rospy.Rate(10)
start_time = time.time()
step = 0
collisions = 0
collision_steps = 0
in_collision = False

print(f"Test running ({duration}s on map_3, respawn after collision)...")
print(f"{'Step':>6} | {'Collisions':>10} | {'Time':>6}")

while not rospy.is_shutdown() and (time.time() - start_time) < duration:
    action = agent.act(state)
    command_manager.step(action)

    next_state = np.reshape(status_reporter.get_state(), [1, state_size])
    collided = status_reporter.is_collided()

    if collided:
        collision_steps += 1
        if not in_collision:
            collisions += 1
            simulation_manager.respawn(parallel_spawn=True)
            time.sleep(0.6)
            next_state = np.reshape(status_reporter.get_state(), [1, state_size])
    in_collision = collided

    state = next_state
    step += 1

    if step % 500 == 0:
        elapsed = time.time() - start_time
        print(f"{step:6} | {collisions:10} | {elapsed:6.0f}s")

    rate.sleep()

elapsed = time.time() - start_time
pct_collision = (collision_steps / step * 100) if step > 0 else 0
collision_free = step - collision_steps

# --- terminal output ---
hline = '=' * 52
print(f"\n{hline}")
print(f"  TEST RESULTS (5 min on map_3)")
print(f"{hline}")
print(f"  Model                 {os.path.basename(model_path)}")
print(f"  Duration              {elapsed:.1f}s")
print(f"  Total steps           {step}")
print(f"  Collisions            {collisions}")
print(f"  Collision steps       {collision_steps:>5} ({pct_collision:.1f}%)")
print(f"  Collision-free steps  {collision_free:>5} ({100-pct_collision:.1f}%)")
if collisions > 0:
    print(f"  Avg steps / collision {step / (collisions + 1):.0f}")
print(f"{hline}")

# --- save report ---
metrics_dir = os.path.join(repo_src, 'metrics')
os.makedirs(metrics_dir, exist_ok=True)
report_path = os.path.join(
    metrics_dir,
    f"test_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
)

report = (
    f"Test Report\n"
    f"{hline}\n"
    f"  Date                  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"  Model                 {os.path.basename(model_path)}\n"
    f"  Map                   map_3\n"
    f"  Duration              {elapsed:.1f}s\n"
    f"  Total steps           {step}\n"
    f"  Collisions            {collisions}\n"
    f"  Collision steps       {collision_steps:>5} ({pct_collision:.1f}%)\n"
    f"  Collision-free steps  {collision_free:>5} ({100-pct_collision:.1f}%)\n"
)
if collisions > 0:
    report += f"  Avg steps / collision  {step / (collisions + 1):.0f}\n"
report += f"{hline}\n"

with open(report_path, 'w') as f:
    f.write(report)
print(f"Report saved: {report_path}")
