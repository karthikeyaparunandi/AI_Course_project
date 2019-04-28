from __future__ import division

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import tensorflow as tf
import deepq  as deepq
from atari_wrappers import wrap_deepmind
#import bench
import logger

env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
env = wrap_deepmind(env, max_and_skip_frame=True, frame_stack=True)


act = deepq.learn(env, network='cnn', total_timesteps=0, buffer_size=0000, exploration_fraction=0.3, dueling=True, target_network_update_freq=500,\
                    gamma=0.99, load_path='experiments/mario_model.pkl')

while True:
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        env.render()
        obs, rew, done, _ = env.step(act(obs[None])[0])
        episode_rew += rew
    print("Episode reward", episode_rew)


