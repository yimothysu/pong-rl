"""
Runs pong.
"""

import argparse

import gym
import numpy as np
import torch
from torchtyping import TensorType
from tqdm import tqdm

from policy import Policy
from preprocess import preprocess


env = gym.make("ALE/Pong-v5", full_action_space=False)

obs_dim = preprocess(np.zeros((env.observation_space.shape))).numel()
act_dim = 3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_trajectory(policy: Policy):
    losses = 0
    wins = 0

    terminated, truncated = False, False

    observations = []
    actions = []
    rewards = []

    observation, info = env.reset()
    prev_observation = preprocess(observation)
    observation = preprocess(observation)
    while not terminated and not truncated:
        diff = (observation - prev_observation).to(device=device)
        action = policy.act(diff.flatten())
        env_action = [0, 2, 3][action]

        observations.append(diff)
        actions.append(action)

        prev_observation = observation
        observation, reward, terminated, truncated, _ = env.step(env_action)
        observation = preprocess(observation)
        if reward > 0:
            wins += 1
            reward *= 5
        if reward < 0:
            losses += 1

        rewards.append(reward)

    print(f"Wins: {wins}, Losses: {losses}")

    return observations, actions, rewards


def collect_trajectories(policy: Policy, N: int = 10) -> tuple[
    TensorType["N", "T", "H", "W", "C"],
    TensorType["N", "T"],
    TensorType["N", "T"],
]:
    """
    Collects `N` trajectories from the environment under the specified policy.

    Returns tensor of shape (N, 3, H), where
    - N is the number of trajectories
    - 3 is the number of elements in the tuple (states, actions, rewards)
    - H is the length of the trajectory
    """
    trajectories = [collect_trajectory(policy) for _ in tqdm(range(N))]
    observations = [trajectory[0] for trajectory in trajectories]
    actions = [trajectory[1] for trajectory in trajectories]
    rewards = [trajectory[2] for trajectory in trajectories]

    return observations, actions, rewards


def train(policy: Policy, epochs: int = 20):
    for epoch in range(epochs):
        print("Epoch", epoch)
        trajectories = collect_trajectories(policy)
        policy.train(trajectories)
    policy.save("model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--epochs", type=int, default=20, required=False)
    args = parser.parse_args()

    policy = Policy(obs_dim, act_dim)
    policy.model.to(device)

    if args.load:
        print("Loading model...")
        policy.load("model.pt")
    print(f"Training for {args.epochs} epochs...")

    train(policy, args.epochs)
    env.close()
