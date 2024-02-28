"""
Runs pong.
"""

import argparse

import gym
import numpy as np
import torch
from torchtyping import TensorType

from policy import Policy
from preprocess import preprocess, preprocess_batch

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect_trajectories(policy: Policy, envs: gym.vector.VectorEnv, N: int = 30, H: int = 1000):
    # store state, action, and reward at each time step for all batches
    observations, actions, rewards = [], [], []

    # reset all environments and process the first batch
    observations_batch, _ = envs.reset()  # shape: (num_envs, height, width, color_channels) = (N, 210, 160, 3)

    prev_observations_batch = preprocess_batch(observations_batch)
    observations_batch = preprocess_batch(observations_batch)
    # prev_observations_batch = torch.zeros_like(observations_batch)  # NOTE: consider blank init prev batch

    for _ in range(H): 
        # get the change in state between observations
        deltas = observations_batch - prev_observations_batch  # (N, 80, 80)

        # get the action for each each flattened observation
        actions_batch = policy.act(deltas.view(N, -1))  # deltas.shape: (N, width*height) = (N, 6400), actions_batch.shape: (N)
        
        # then map these to actions 0 - NOOP, 2 - RIGHT, 3 - LEFT: [0, 1, 2] -> [0, 2, 3]
        # envs_actions = [[0, 2, 3][action] for action in actions_batch] # envs.steps does not accept tensors
        envs_actions = np.array([0, 2, 3])[actions_batch]  # (N)

        # save the previous batch and check for end conditions
        prev_observations_batch = observations_batch

        # take a step in every environment
        observations_batch, rewards_batch, terminated, truncated, _ = envs.step(envs_actions)

        # preprocess the observation for the network
        observations_batch = preprocess_batch(observations_batch)  # (N, 80, 80)

        # save current episode data
        observations.append(deltas)    # save the change in the reduced-state observation
        actions.append(actions_batch)  # NOTE: following train.py, this stores actions, not envs_actions
        rewards.append(rewards_batch)

        # TODO: make sure this is working as intended :)
        if np.all(terminated) or np.all(truncated):
            break
    
    # convert the lists of np arrays to tensors
    observations = torch.stack([torch.tensor(obs) for obs in observations]) # (H, N, height, width) = (H, N, 80, 80)
    actions = torch.stack([torch.tensor(act) for act in actions])           # (H, N)
    rewards = torch.stack([torch.tensor(rew) for rew in rewards])           # (H, N)

    # transpose the dimensions to match train.py
    observations = observations.transpose(0, 1)  #  [N, H, 80, 80]
    actions = actions.transpose(0, 1)            #  [N, H]
    rewards = rewards.transpose(0, 1)            #  [N, H]

    return observations, actions, rewards

        
    '''
    800 -- horizon
    30 -- num epochs
    80,80 -- reduced state space


    torch.Size([30, 800, 80, 80])
    torch.Size([30, 800])
    torch.Size([30, 800])

    when this works, there needs to be zero padding from the time the trajectory
    stops, T, to the horizon H
            T1       H
    | 1, 1, 1, 0, 0, 0 |
    | 1, 1, 1, 1, 1, 1 |
    
    '''



def train(policy: Policy, envs: gym.vector.VectorEnv, epochs: int = 20, H: int = 1000):
    start_total = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start_epoch = time.time()
        trajectories = collect_trajectories(policy, envs=envs, N=envs.num_envs, H=1000)
        end_epoch = time.time()
        print(f" - Time to collect trajectories: {end_epoch-start_epoch:.2f} seconds")
        policy.train(trajectories)
        print(30*"-")
        print()
    total_time = time.time() - start_total
    print(f"Total training time: {total_time:.2f} seconds\n")
    policy.save("model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--epochs", type=int, default=20, required=False)
    parser.add_argument("--horizon", type=int, default=1000, required=False)
    parser.add_argument("--envs", type=int, default=30, help="Number of parallel environments.", required=False)
    args = parser.parse_args()

    envs = gym.vector.make("ALE/Pong-v5", num_envs=args.envs, full_action_space=False)

    obs_dim = preprocess(np.zeros((envs.single_observation_space.shape))).numel()
    print(obs_dim)
    act_dim = 3

    policy = Policy(obs_dim=obs_dim, act_dim=act_dim) 
    policy.model.to(device)

    if args.load:
        print("Loading model...")
        policy.load("model.pt")

    print(f"Training for {args.epochs} epochs with {args.envs} parallel environments:")

    train(policy, envs=envs, epochs=args.epochs, H=args.horizon)  

    envs.close()  