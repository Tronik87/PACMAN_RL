import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import numpy as np

def make_env():
    env = gym.make("ALE/MsPacman-v5")
    env = AtariWrapper(env)
    return env

if __name__ == "__main__":
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    
    print("Loading baseline model from baseline_pacman.zip...")
    try:
        model = DQN.load("baseline_pacman", env=env)
    except Exception as e:
        print(f"Failed to load baseline: {e}")
        exit(1)
        
    MAX_STEPS_PER_EVAL = 2000
    NUM_EPISODES = 5
    
    print(f"Evaluating over {NUM_EPISODES} episodes (max {MAX_STEPS_PER_EVAL} steps each)...")
    fitness_scores = []
    
    for ep in range(NUM_EPISODES):
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < MAX_STEPS_PER_EVAL:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            steps += 1
            
            if done[0]:
                break
                
        print(f"Episode {ep+1} Fitness (Total Reward): {total_reward}")
        fitness_scores.append(total_reward)
        
    with open("baseline_fitness.txt", "w") as f:
        f.write("-------------------------------------------------\n")
        f.write(f"Baseline DDQN Avg Fitness:   {np.mean(fitness_scores)}\n")
        f.write(f"Baseline DDQN Max Fitness:   {np.max(fitness_scores)}\n")
        
    print("Saved baseline fitness metrics to baseline_fitness.txt")
