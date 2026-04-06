import torch
import numpy as np
import pickle
import os
import csv
from environment import create_env
from stable_baseline_model import create_ddqn
from ga_model import get_weights_as_1d, set_weights_from_1d

# ----- GA Hyperparameters -----
POPULATION_SIZE = 15
GENERATIONS = 50
ELITISM = 3  # Keep top 3 exactly as is
MUTATION_RATE = 0.05 # Probability of mutating any given weight
MUTATION_POWER = 0.05 # Standard deviation of gaussian noise
MAX_STEPS_PER_EVAL = 2000 # To prevent going on forever if stuck

# Ensure logs dir exists
os.makedirs("logs", exist_ok=True)

# 1. Initialize Single Model & Env
print("Initializing environment and base model...")
env = create_env()
# A Dummy VecEnv with VecFrameStack
# We pass this env into create_ddqn to let it structure the network correctly
model = create_ddqn(env)

# Extract PyTorch Policy's Q-network
q_net = model.policy.q_net

def evaluate_fitness(weights_1d):
    """
    Evaluates fitness by playing one episode.
    Uses clipped rewards (standard for Atari DQNs).
    """
    set_weights_from_1d(q_net, weights_1d)
    
    obs = env.reset()
    total_reward = 0
    steps = 0
    
    while steps < MAX_STEPS_PER_EVAL:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # reward is an array because of dummyVecEnv (clipped by AtariWrapper)
        total_reward += float(reward[0])
        steps += 1
        
        if done[0]:
            break
            
    return total_reward

if __name__ == "__main__":
    # 2. Initialize Population
    print("Initializing population...")
    
    # Load best weights if they exist, otherwise use DDQN base weights
    best_weights_path = "ga_best_pacman.pt"
    if os.path.exists(best_weights_path):
        print(f" -> Loading previous best weights from {best_weights_path}")
        base_weights = torch.load(best_weights_path)
    else:
        print(" -> Starting with fresh DDQN weights.")
        base_weights = get_weights_as_1d(q_net)
        
    vec_length = len(base_weights)

    # Create an initial population around the base weights
    population = []
    # Seed individual 0 as the pure base weights
    population.append(base_weights.clone())
    for _ in range(POPULATION_SIZE - 1):
        noise = torch.randn(vec_length) * MUTATION_POWER
        new_weights = base_weights + noise
        population.append(new_weights)

    stats = {"max": [], "avg": []}

    try:
        best_overall_fitness = -float("inf")
        best_overall_weights = None
        # Prepare CSV file and detect starting generation
        csv_path = "logs/ga_training_results.csv"
        start_gen = 0
        
        if os.path.exists(csv_path):
            with open(csv_path, mode='r') as f:
                lines = list(csv.reader(f))
                if len(lines) > 1:
                    try:
                        start_gen = int(lines[-1][0])
                        print(f" -> Resuming from generation {start_gen}")
                    except (ValueError, IndexError):
                        pass
        
        # Write headers only if new file
        if start_gen == 0:
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["generation", "max_fitness", "avg_fitness", "min_fitness"])

        for gen_idx in range(GENERATIONS):
            gen = start_gen + gen_idx + 1
            print(f"\n--- Generation {gen} ---")
            
            # Evaluate All
            fitness_scores = []
            for i, ind in enumerate(population):
                fit = evaluate_fitness(ind)
                fitness_scores.append(fit)
                print(f" Individual {i}: Fitness = {fit}")
                
            # Get Stats
            max_fit = np.max(fitness_scores)
            avg_fit = np.mean(fitness_scores)
            stats["max"].append(max_fit)
            stats["avg"].append(avg_fit)
            
            print(f"Gen {gen} Summary: Max: {max_fit}, Avg: {avg_fit:.2f}")
            
            # Check if absolute best
            if max_fit > best_overall_fitness:
                best_overall_fitness = max_fit
                best_idx = np.argmax(fitness_scores)
                best_overall_weights = population[best_idx].clone()
                torch.save(best_overall_weights, "ga_best_pacman.pt")
                print(" -> New Best Model Saved!")

            # Log to CSV
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([gen, max_fit, avg_fit, np.min(fitness_scores)])

            # Sort population by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1] # descending
            sorted_pop = [population[i] for i in sorted_indices]
            
            # Next Generation
            new_population = []
            
            # Elitism
            for i in range(ELITISM):
                new_population.append(sorted_pop[i].clone())
                
            # Select parents (from top half)
            top_half = sorted_pop[:POPULATION_SIZE // 2]
            
            while len(new_population) < POPULATION_SIZE:
                # Crossover
                parent1_idx = np.random.randint(len(top_half))
                parent2_idx = np.random.randint(len(top_half))
                parent1 = top_half[parent1_idx]
                parent2 = top_half[parent2_idx]
                
                # Uniform crossover
                mask = torch.rand(vec_length) > 0.5
                child = torch.where(mask, parent1, parent2)
                
                # Mutation
                mutation_mask = torch.rand(vec_length) < MUTATION_RATE
                mutation_noise = torch.randn(vec_length) * MUTATION_POWER
                child = child + (mutation_mask * mutation_noise)
                
                new_population.append(child)
                
            population = new_population
            
    except KeyboardInterrupt:
        print("\nTraining interrupted manually.")

    # Save stats
    with open("logs/ga_stats.pkl", "wb") as f:
        pickle.dump(stats, f)
    print("Saved stats to logs/ga_stats.pkl")

    print(f"Best overall fitness achieved: {best_overall_fitness}")
