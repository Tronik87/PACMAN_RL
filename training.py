import os
from environment import create_env
from stable_baseline_model import create_ddqn
from stable_baselines3.common.callbacks import BaseCallback

class TrainLoggingCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(TrainLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'ddqn_training_results.csv')

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(self.save_path, 'w') as f:
                f.write("timestep,episode_reward\n")

    def _on_step(self) -> bool:
        # Check if an episode finished
        if 'episode' in self.locals['infos'][0]:
            reward = self.locals['infos'][0]['episode']['r']
            timestep = self.num_timesteps
            with open(self.save_path, 'a') as f:
                f.write(f"{timestep},{reward}\n")
        return True

env=create_env()
model=create_ddqn(env)

# Create callback
callback = TrainLoggingCallback(check_freq=1000, log_dir="./logs")

print("Starting DDQN training (Comparison Run)...")
new_model_name = "ddqn_comparison_model"

try:
    model.learn(total_timesteps=200000, callback=callback)
    print("\nDDQN training reached full time limit.")
except KeyboardInterrupt:
    print("\nTraining interrupted! Saving current progress...")

# Save to the new filename to protect your original baseline
model.save(new_model_name)
print(f"Model saved as '{new_model_name}.zip'. Original 'baseline_pacman.zip' remains untouched.")