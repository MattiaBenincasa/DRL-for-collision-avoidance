#!/usr/bin/env python3
"""
Training script implementing a Double DQN aligned to Feng et al., Robotics 2021, 10, 73 (doi:10.3390/robotics10020073).
Key mappings: state size = 52 (50 LIDAR + 2 goal info), action space = 11 network: two hidden layers (256, 128), γ=0.99, ε-greedy schedule decays to 0.05.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(2)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DDQNAgent:
    def __init__(self, state_size, action_size, reward_func):
        self.state_size = state_size
        self.action_size = action_size
        self.reward_func = reward_func
        self.max_memory = 30000
        self.ptr = 0
        self.count = 0
        self.states      = np.zeros((self.max_memory, self.state_size), dtype=np.float32)
        self.actions     = np.zeros(self.max_memory, dtype=np.int32)
        self.rewards     = np.zeros(self.max_memory, dtype=np.float32)
        self.next_states = np.zeros((self.max_memory, self.state_size), dtype=np.float32)
        self.dones       = np.zeros(self.max_memory, dtype=np.bool_)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.update_target_freq = 1000

        self.batch_size = 64
        self.eval_model = self._build_model()
        self.target_model = self._build_model()
        # initialize target network weights
        self.target_model.set_weights(self.eval_model.get_weights())
        self.steps_since_target_update = 0  # Counter for target network updates

    def _build_model(self):
        model = Sequential()
        model.add(Dense(300, input_dim=self.state_size, activation='relu'))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.eval_model.predict_on_batch(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx]      = state.reshape(-1)
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.next_states[idx] = next_state.reshape(-1)
        self.dones[idx]       = done
        self.ptr = (self.ptr + 1) % self.max_memory
        self.count = min(self.count + 1, self.max_memory)

    # Double DQN update: select action with eval network and evaluate with target network to reduce overestimation (see Feng et al., Robotics 2021).
    def replay(self):
        if self.count < self.batch_size:
            return

        indices = np.random.choice(self.count, self.batch_size, replace=False)
        states      = self.states[indices]      # (B, state_size)
        actions     = self.actions[indices]     # (B,)
        rewards     = self.rewards[indices]     # (B,)
        next_states = self.next_states[indices] # (B, state_size)
        dones       = self.dones[indices]       # (B,)

        # Q(s,·) current
        q_values = self.eval_model.predict_on_batch(states)               # (B, action_size)

        # Double DQN:
        # a* = argmax_a Q_eval(s', a)
        q_next_eval = self.eval_model.predict_on_batch(next_states)       # (B, action_size)
        next_actions = np.argmax(q_next_eval, axis=1)                     # (B,)

        # Q_target(s', a*)
        q_next_target = self.target_model.predict_on_batch(next_states)   # (B, action_size)
        q_next = q_next_target[np.arange(self.batch_size), next_actions]   # (B,)

        targets = rewards + self.gamma * q_next * (~dones)
        # update only the executed action
        q_values[np.arange(self.batch_size), actions] = targets

        self.eval_model.train_on_batch(states, q_values)

    def on_step(self):
        self.steps_since_target_update += 1
        if self.steps_since_target_update >= self.update_target_freq:
            self.update_target_network()
            self.steps_since_target_update = 0

    def update_target_network(self):
        """Update target network weights from eval network (infrequent)."""
        self.target_model.set_weights(self.eval_model.get_weights())

    def get_reward(self, collided, **kwargs):
        """Computes reward by delegating to self.reward_func.

        Available kwargs (passed by the training loop):
            robot_pose:   Pose2D with .x, .y, .angle_deg — current position
            pos_prev:     Pose2D — position at previous step (None at step 0)
            state:        numpy array (1, state_size) — [50 LIDAR norm, dist, angle]
            next_state:   numpy array (1, state_size) — state after action
            steps:        int — current step in the episode
            total_reward: float — accumulated reward in the episode
            action:       int — index of the executed action
            step_limit:   int — episode timeout in steps
            target_info:  tuple (cx, cy, sx, sy, yaw) — full target zone
        """
        return self.reward_func(collided, **kwargs)

    # Save trained model to .h5 file
    def save(self, name):
        self.eval_model.save(name)
