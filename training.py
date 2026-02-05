from environment import create_env
from stable_baseline_model import create_ddqn

env=create_env()
model=create_ddqn(env)

model.learn(total_timesteps=200000)
model.save("baseline_pacman")