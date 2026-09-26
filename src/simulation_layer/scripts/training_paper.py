#!/usr/bin/env python3

import os
import sys

# Add workspace src to PYTHONPATH so package imports work when workspace is not sourced
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
import time
import numpy as np
import gc

rospy.init_node('storm_training_node')

simulation_manager = SimulationManager('map_2')
sim_position_tracker = SimPositionTracker(simulation_manager)
laser_sensor = LaserSensor()
status_reporter = StatusReporter(laser_sensor=laser_sensor)
command_manager = CommandManager(status_reporter)

state_size = 50
action_size = 11

def default_reward(collided, **kwargs):
    if collided:
        return -1000, True
    return 5, False

agent = DDQNAgent(state_size, action_size, reward_func=default_reward)
episodes = 3000

model_path = os.path.join(repo_src, 'undrafted_models', 'ddqn_model.keras')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

metrics_dir = os.path.join(repo_src, "metrics")
metrics_path = os.path.join(metrics_dir, "ddqn_metrics_episodes.csv")
# Build config to be saved as metadata
config = {
    "episodes": episodes,
    "state_size": state_size,
    "action_size": action_size,
    "model_path": model_path,
}
# Collect agent hyperparameters if available
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
        "spawn_id"
    ],
    append=True,
    config=config,
)

for e in range(episodes):
    simulation_manager.respawn(parallel_spawn=True)
    spawn = simulation_manager.last_spawn
    sim_position_tracker.set_start_position(spawn)
    time.sleep(0.3)

    state = np.reshape(status_reporter.get_state(), [1, state_size])

    done = False

    steps = 0
    total_reward = 0.0
    collided = False

    rate = rospy.Rate(10)
    # The episode ends when the robot collides or time limit is reached.
    while not done and not rospy.is_shutdown():
        action = agent.act(state)
        command_manager.step(action)

        next_state = np.reshape(status_reporter.get_state(), [1, state_size])

        # get robot pose once per loop
        robot_pose = simulation_manager.get_position("storm")
        collided = status_reporter.is_collided()

        reward, done = agent.get_reward(
            collided,
            robot_pose=robot_pose,
            state=state,
            next_state=next_state,
            steps=steps,
            total_reward=total_reward,
            action=action,
        )

        agent.remember(state, action, reward, next_state, done)
        agent.on_step()

        state = next_state

        steps += 1

        if steps % 8 == 0:
            simulation_manager.pause_simulation()
            try:
                    agent.replay()
            finally:
                simulation_manager.resume_simulation()

        total_reward += float(reward)

        if steps >= simulation_manager.step_limit:
            done = True

        rate.sleep()

    if rospy.is_shutdown():
        agent.save(model_path)
        break

    spawn_id=getattr(getattr(simulation_manager, 'last_spawn', None), 'name', None)

    # Log ONE row per episode
    sm.add(
        episode=int(e),
        steps=int(steps),
        total_reward=float(total_reward),
        epsilon=float(agent.epsilon),
        collided=bool(collided),
        spawn_id=spawn_id
    )
    sm.write_episode()

    print(
        f"Episode: {e}/{episodes} | "
        f"RewardTot: {total_reward:.3f} | "
        f"Steps: {steps} | "
        f"Epsilon: {agent.epsilon:.3f} | "
        f"Collided: {collided} | "
        f"Spawn: {spawn_id} | "
    )

    if e % 10 == 0:
        agent.save(model_path)

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if e % 50 == 0:
        gc.collect()

agent.save(model_path)
