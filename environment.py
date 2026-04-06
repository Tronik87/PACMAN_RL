import gymnasium as gym
import ale_py

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

def make_env():
    env=gym.make("ALE/MsPacman-v5")
    env=Monitor(env)
    env=AtariWrapper(env)
    return env

def create_env():
    env=DummyVecEnv([make_env])
    env=VecFrameStack(env , n_stack=4)
    return env
