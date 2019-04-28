from __future__ import division

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from PIL import Image
import numpy as np

import tensorflow as tf
from baselines import deepq
from baselines.common.atari_wrappers import wrap_deepmind
from baselines import bench
from baselines import logger

env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
env = wrap_deepmind(env, max_and_skip_frame=True, episode_life=False, frame_stack=True, clip_rewards=False)
logger.configure()

env = bench.Monitor(env, logger.get_dir())



act = deepq.learn(env, network='cnn', total_timesteps=1000000, buffer_size=100000, exploration_fraction=0.3, prioritized_replay=True, dueling=True, target_network_update_freq=500,\
					gamma=0.99, exploration_final_eps=0.02,	param_noise=False, print_freq=10)
print("Saving model to mario_model.pkl")
act.save("mario_model.pkl")


