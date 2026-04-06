import torch
import gymnasium as gym
import ale_py
from stable_baseline_model import create_ddqn
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from ga_model import set_weights_from_1d

def make_env_render():
    env=gym.make("ALE/MsPacman-v5", render_mode="human")
    env=AtariWrapper(env)
    return env

if __name__ == "__main__":
    env = DummyVecEnv([make_env_render])
    env = VecFrameStack(env, n_stack=4)

    print("Initializing architecture...")
    model = create_ddqn(env)
    q_net = model.policy.q_net

    try:
        print("Loading best GA weights...")
        best_weights = torch.load("ga_best_pacman.pt")
        set_weights_from_1d(q_net, best_weights)
    except FileNotFoundError:
        print("Error: 'ga_best_pacman.pt' not found.")
        exit(1)

    obs = env.reset()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
    except KeyboardInterrupt:
        print("Stopped playing.")
