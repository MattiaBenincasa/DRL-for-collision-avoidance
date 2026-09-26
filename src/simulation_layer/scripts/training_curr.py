#!/usr/bin/env python3

import os
import sys

repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

import rospy
from control_layer.scripts.DDQNAgent import DDQNAgent
from platform_layer.scripts.CommandManager import CommandManager
from platform_layer.scripts.StatusReporter import StatusReporter
from platform_layer.scripts.LaserSensor import LaserSensor
from platform_layer.scripts.SimPositionTracker import SimPositionTracker
from simulation_layer.scripts.SimulationManager import SimulationManager
from simulation_layer.scripts.SaveMetrics import SaveMetrics
from simulation_layer.scripts.CurriculumManager import CurriculumManager
from simulation_layer.scripts.Pose2D import Pose2D
import time
import numpy as np
import math

rospy.init_node('storm_curriculum_training_node')

simulation_manager = SimulationManager('map_curr')
sim_position_tracker = SimPositionTracker(simulation_manager)
laser_sensor = LaserSensor()
status_reporter = StatusReporter(laser_sensor=laser_sensor, state_size=50)
command_manager = CommandManager(status_reporter)

curriculum = CurriculumManager(max_level=4, window_size=20,
                               advance_threshold=0.5, regress_threshold=0.3,
                               consecutive_fails_for_regress=10)

state_size = 52
action_size = 11
LEVEL_STEP_LIMITS = {1: 300, 2: 500, 3: 750, 4: 1000}


def _in_target_zone(px, py, target_info):
    cx, cy, sx, sy, yaw = target_info
    dx = px - cx
    dy = py - cy
    cosy = math.cos(-yaw)
    siny = math.sin(-yaw)
    lx = dx * cosy - dy * siny
    ly = dx * siny + dy * cosy
    return -sx/2 <= lx <= sx/2 and -sy/2 <= ly <= sy/2

def default_reward(collided, robot_pose=None, pos_prev=None, steps=None,
                   step_limit=None, target_info=None, **kwargs):
    if collided:
        return -200, True

    if steps >= step_limit:
        return -50, True

    cx, cy = target_info[0], target_info[1]
    dist_curr = math.hypot(robot_pose.x - cx, robot_pose.y - cy)
    dist_prev = math.hypot(pos_prev.x - cx, pos_prev.y - cy)

    if _in_target_zone(robot_pose.x, robot_pose.y, target_info):
        return 500, True

    delta = dist_prev - dist_curr
    progress_reward = 50.0 * delta

    movement_bonus = 5.0 if delta > 0.05 else 0.0
    time_penalty = -0.05

    return progress_reward + movement_bonus + time_penalty, False

agent = DDQNAgent(state_size, action_size, reward_func=default_reward)
# Replace default_reward with your own custom reward function
episodes = 3000

model_path = os.path.join(repo_src, 'undrafted_models', 'ddqn_curr_model.h5')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

metrics_dir = os.path.join(repo_src, "metrics")
metrics_path = os.path.join(metrics_dir, "ddqn_curr_metrics_episodes.csv")
config = {
    "episodes": episodes,
    "state_size": state_size,
    "action_size": action_size,
    "model_path": model_path,
    "curriculum": {
        "max_level": curriculum.max_level,
        "window_size": curriculum.window.maxlen,
        "advance_threshold": curriculum.advance_threshold,
        "regress_threshold": curriculum.regress_threshold,
        "consecutive_fails_for_regress": curriculum.consecutive_fails_for_regress,
        "min_episodes_at_level": curriculum.min_episodes_at_level,
    },
}
try:
    config["agent"] = {
        "memory_maxlen": getattr(agent.memory, "maxlen", None),
        "gamma": getattr(agent, "gamma", None),
        "epsilon_start": getattr(agent, "epsilon", None),
        "epsilon_min": getattr(agent, "epsilon_min", None),
        "epsilon_decay": getattr(agent, "epsilon_decay", None),
        "batch_size": getattr(agent, "batch_size", None),
    }
    try:
        lr = agent.eval_model.optimizer.learning_rate
        try:
            config["agent"]["learning_rate"] = float(lr)
        except Exception:
            config["agent"]["learning_rate"] = str(lr)
    except Exception:
        config["agent"]["learning_rate"] = None
except Exception:
    pass

sm = SaveMetrics(
    csv_path=metrics_path,
    fieldnames=[
        "episode",
        "steps",
        "total_reward",
        "epsilon",
        "collided",
        "spawn_id",
        "curriculum_level",
        "reached_target",
    ],
    append=True,
    config=config,
)

for e in range(episodes):
    simulation_manager.respawn(parallel_spawn=False, spawn_index=curriculum.spawn_index)
    spawn = simulation_manager.last_spawn
    sim_position_tracker.set_start_position(spawn)
    time.sleep(0.3)

    simulation_manager.step_limit = LEVEL_STEP_LIMITS.get(curriculum.current_level, 1000)

    map_info = simulation_manager.maps.get(simulation_manager.current_map)
    if map_info and map_info.targets:
        tz = next(iter(map_info.targets.values()))
        target_info = (tz.x, tz.y, tz.sx or 1.0, tz.sy or 1.0, tz.yaw)
    else:
        target_info = (0.0, 0.0, 1.0, 1.0, 0.0)

    pos_prev = Pose2D(spawn.x, spawn.y, spawn.yaw)
    pos_curr = pos_prev

    state = np.reshape(status_reporter.get_state(robot_pose=pos_curr, target_pose=target_info[:2]), [1, state_size])

    done = False

    steps = 0
    total_reward = 0.0
    collided = False
    reached_target = False

    rate = rospy.Rate(10)
    while not done and not rospy.is_shutdown():
        action = agent.act(state)
        command_manager.step(action)

        next_state = np.reshape(status_reporter.get_state(robot_pose=pos_curr, target_pose=target_info[:2]), [1, state_size])

        robot_pose = simulation_manager.get_position("storm")
        if robot_pose is not None:
            pos_prev = pos_curr
            pos_curr = robot_pose
        collided = status_reporter.is_collided()

        reward, done = agent.get_reward(
            collided,
            robot_pose=pos_curr,
            pos_prev=pos_prev,
            state=state,
            next_state=next_state,
            steps=steps,
            total_reward=total_reward,
            action=action,
            step_limit=simulation_manager.step_limit,
            target_info=target_info,
        )

        agent.remember(state, action, reward, next_state, done)
        agent.on_step()

        if not collided and _in_target_zone(pos_curr.x, pos_curr.y, target_info):
            reached_target = True

        total_reward += float(reward)

        state = next_state

        steps += 1

        if steps % 4 == 0:
            simulation_manager.pause_simulation()
            try:
                agent.replay()
            finally:
                simulation_manager.resume_simulation()

        if steps >= simulation_manager.step_limit:
            done = True

        rate.sleep()

    if rospy.is_shutdown():
        agent.save(model_path)
        break

    spawn_id = getattr(getattr(simulation_manager, 'last_spawn', None), 'name', None)

    curriculum.update(success=reached_target)

    sm.add(
        episode=int(e),
        steps=int(steps),
        total_reward=float(total_reward),
        epsilon=float(agent.epsilon),
        collided=bool(collided),
        spawn_id=spawn_id,
        curriculum_level=int(curriculum.current_level),
        reached_target=bool(reached_target),
    )
    sm.write_episode()

    print(
        f"Episode: {e}/{episodes} | "
        f"RewardTot: {total_reward:.3f} | "
        f"Steps: {steps} | "
        f"Epsilon: {agent.epsilon:.3f} | "
        f"Collided: {collided} | "
        f"Target: {reached_target} | "
        f"Spawn: {spawn_id} | "
        f"Level: {curriculum.current_level} | "
    )

    if e % 10 == 0:
        agent.save(model_path)

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

agent.save(model_path)
