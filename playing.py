import gymnasium as gym
import ale_py

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def make_env_render():
    env=gym.make("ALE/MsPacman-v5", render_mode="human")
    env=AtariWrapper(env)
    return env

env=DummyVecEnv([make_env_render])
env=VecFrameStack(env, n_stack=4)

model=DQN.load("baseline_pacman", env=env)
obs=env.reset()

while True:
    action,_=model.predict(obs , deterministic=True)
    obs, reward , done , info=env.step(action)
    
    