from stable_baselines3 import DQN

def create_ddqn(env):
    model=DQN(
        "CnnPolicy",
        env, 
        verbose=1,
        learning_rate=0.0001,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=32,
        gamma=0.99,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        tensorboard_log="./logs"
    )
    
    '''verbose: for logs
                0: no logs
                1: basic info
                2: detailed logs
        buffer_size: size of the replay buffer
        learning_starts: how many experiences to collect 
            before learning starts
        batch_size: no of experiences sampled per training update
        gamma: discount factor- how much future rewards matter
        target_update_interval: how often weights are copied
            from online to target network
        exploration_fraction: controls epsilon-greedy decay
        exploration_final_eps: minimum exploration rate
        '''
    return model