import torch
import torch.nn as nn
import numpy as np
import copy
from environment import create_env

# 1. Define the Neural Network Architecture
class PacmanCNN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(PacmanCNN, self).__init__()
        # Standard CNN Architecture for Atari Games
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Neural Networks prefer normalized inputs [0, 1] instead of [0, 255]
        x = x / 255.0 
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # Flatten the 3D tensor to 1D
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 2. Genetic Algorithm Functions
def mutate(model, mutation_rate=0.1, mutation_strength=0.1):
    """Randomly mutates the weights of the model."""
    with torch.no_grad():
        for param in model.parameters():
            if torch.rand(1).item() < mutation_rate:
                # Add gaussian noise to the weights to simulate mutations
                noise = torch.randn(param.size()) * mutation_strength
                param.add_(noise)

def crossover(parent1, parent2):
    """Combines the "genes" (weights) of two parent networks."""
    child = copy.deepcopy(parent1)
    with torch.no_grad():
        for child_param, p2_param in zip(child.parameters(), parent2.parameters()):
            # ~50% chance for each weight to be taken from parent 2
            mask = torch.rand(child_param.size()) > 0.5
            child_param[mask] = p2_param[mask]
    return child

# 3. Environment Evaluation Loop
def evaluate(agent, env, max_steps=1500):
    """Runs one episode in the environment the returns the fitness score (reward)."""
    obs = env.reset()
    total_reward = 0
    with torch.no_grad():
        for _ in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            
            # The environment returns (Batch=1, Height=84, Width=84, Channels=4)
            # PyTorch expects (Batch=1, Channels=4, Height=84, Width=84)
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
            
            # Get Q-Values and take the highest one
            q_values = agent(obs_tensor)
            action = [torch.argmax(q_values, dim=1).item()]
            
            obs, reward, done, info = env.step(action)
            total_reward += reward[0] # environment returns arrays because it's vectorized
            
            if done[0]:
                break
    return total_reward

# 4. Main Training Loop
if __name__ == "__main__":
    POPULATION_SIZE = 10
    GENERATIONS = 10
    MUTATION_RATE = 0.2
    
    print("Setting up MsPacman Environment...")
    env = create_env()
    
    # In gym environments, discrete action space size is accessed via `n`
    # MsPacman has 9 possible actions
    num_actions = env.action_space.n 
    
    print(f"Initializing a new population of {POPULATION_SIZE} agents...")
    population = [PacmanCNN(4, num_actions) for _ in range(POPULATION_SIZE)]
    
    for generation in range(GENERATIONS):
        print(f"\n--- Generation {generation+1} ---")
        fitness_scores = []
        
        # 1. Evaluate Fitness of all individuals
        for i, agent in enumerate(population):
            score = evaluate(agent, env)
            fitness_scores.append(score)
            print(f"  > Agent {i} Score: {score}")
            
        # 2. Select the top performers (Survival of the fittest)
        # np.argsort sorts smallest to largest; [::-1] reverses it
        sorted_indices = np.argsort(fitness_scores)[::-1] 
        best_agent_idx = sorted_indices[0]
        print(f">> Best Score in Generation {generation+1}: {fitness_scores[best_agent_idx]}")
        
        # We will keep the top 2 parents based on fitness
        parent1 = population[sorted_indices[0]]
        parent2 = population[sorted_indices[1]]
        
        # 3. Create next generation
        # Elitism: Directly pass the top two unmodified to the next gen so we don't regress
        new_population = [copy.deepcopy(parent1), copy.deepcopy(parent2)] 
        
        for _ in range(POPULATION_SIZE - 2):
            # Crossover and Mutate the rest
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate=MUTATION_RATE)
            new_population.append(child)
            
        population = new_population # Replace old population
        
    print("\nGenetic Algorithm Evolution Finished!")
    # Save the absolute best
    torch.save(parent1.state_dict(), "best_ga_pacman.pth")
    print("Saved the best evolved agent's brain as 'best_ga_pacman.pth'.")
