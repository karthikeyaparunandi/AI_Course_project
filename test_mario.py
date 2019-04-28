from __future__ import division

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import tensorflow as tf
import DQN.deepq  as deepq
from DQN.atari_wrappers import wrap_deepmind
import DQN.logger as logger


# creating an environment
env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
env = wrap_deepmind(env, max_and_skip_frame=True, frame_stack=True)

# creating the DQN model with 'Prioritized Experienced Replay' and 'Duel Network'
act = deepq.learn(env, network='cnn', total_timesteps=0, buffer_size=0000, exploration_fraction=0.3, dueling=True, target_network_update_freq=500,\
                    gamma=0.99, load_path='experiments/mario_model.pkl')

while True:
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        env.render()
        obs, reward, done, _ = env.step(act(obs[None])[0])
        episode_rew += reward
    print("Episode reward", episode_rew)


