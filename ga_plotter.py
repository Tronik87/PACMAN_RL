import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def plot_comparison():
    # Paths
    ga_path = "logs/ga_training_results.csv"
    ddqn_path = "logs/ddqn_training_results.csv"
    
    # Check if files exist
    if not os.path.exists(ga_path):
        print(f"Warning: {ga_path} not found. GA data will be missing.")
        ga_df = None
    else:
        ga_df = pd.read_csv(ga_path)
        
    if not os.path.exists(ddqn_path):
        print(f"Warning: {ddqn_path} not found. DDQN data will be missing.")
        ddqn_df = None
    else:
        ddqn_df = pd.read_csv(ddqn_path)

    # Styling
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#121212')
    
    # 1. GA Plot
    if ga_df is not None:
        ax1.plot(ga_df['generation'], ga_df['max_fitness'], label='Max Fitness', color='#00FFCC', linewidth=2, marker='o', markersize=4)
        ax1.plot(ga_df['generation'], ga_df['avg_fitness'], label='Avg Fitness', color='#00CCFF', linestyle='--', alpha=0.8)
        ax1.fill_between(ga_df['generation'], ga_df['min_fitness'], ga_df['max_fitness'], color='#00FFCC', alpha=0.1)
        ax1.set_title("Genetic Algorithm Progress", fontsize=14, fontweight='bold', color='#00FFCC')
        ax1.set_xlabel("Generation", fontsize=12)
        ax1.set_ylabel("Fitness (Reward)", fontsize=12)
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "GA Data Missing\nRun ga_training.py", ha='center', va='center', color='gray')

    # 2. DDQN Plot
    if ddqn_df is not None:
        # Calculate rolling average for DDQN as it's noisy
        window = max(1, len(ddqn_df) // 10)
        ddqn_df['rolling_reward'] = ddqn_df['episode_reward'].rolling(window=window).mean()
        
        ax2.scatter(ddqn_df['timestep'], ddqn_df['episode_reward'], color='#FF3366', alpha=0.2, s=10, label='Raw Reward')
        ax2.plot(ddqn_df['timestep'], ddqn_df['rolling_reward'], color='#FF3366', linewidth=2, label=f'Rolling Avg (w={window})')
        
        ax2.set_title("DDQN Baseline Progress", fontsize=14, fontweight='bold', color='#FF3366')
        ax2.set_xlabel("Timesteps", fontsize=12)
        ax2.set_ylabel("Reward", fontsize=12)
        ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "DDQN Data Missing\nRun training.py", ha='center', va='center', color='gray')

    plt.tight_layout()
    
    # Save results
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/training_comparison.png", dpi=300, facecolor='#121212')
    print("Successfully generated comparison plot: logs/training_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_comparison()
