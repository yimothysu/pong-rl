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


def collect_trajectory(policy: Policy, H: int = 100):
    losses = 0
    wins = 0

    observations = []
    actions = []
    rewards = []

    observation, info = env.reset()
    prev_observation = preprocess(observation)
    observation = preprocess(observation)
    for _ in range(H):
        action = policy.act(
            (observation - prev_observation).flatten().to(device=device)
        )
        env_action = [0, 2, 3][action]
        observation, reward, terminated, truncated, _ = env.step(env_action)
        observation = preprocess(observation)
        if reward > 0:
            wins += 1
        if reward < 0:
            losses += 1

        observations.append(observation - prev_observation)
        actions.append(action)
        rewards.append(reward)

        prev_observation = observation

        if terminated or truncated:
            break

    observations_content = torch.stack(observations, dim=0).to(device=device)
    observations = torch.zeros((H, *observations_content.shape[1:])).to(device=device)
    observations[: observations_content.shape[0]] = observations_content

    actions_content = torch.tensor(actions).to(device=device)
    actions = torch.zeros(H, dtype=torch.int64).to(device=device)
    actions[: actions_content.shape[0]] = actions_content

    rewards_content = torch.tensor(rewards).to(device=device)
    rewards = torch.zeros(H, dtype=torch.float32).to(device=device)
    rewards[: rewards_content.shape[0]] = rewards_content

    print(f"Wins: {wins}, Losses: {losses}")

    return observations, actions, rewards


def collect_trajectories(policy: Policy, N: int = 40, H: int = 100) -> tuple[
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
    trajectories = [collect_trajectory(policy, H) for _ in tqdm(range(N))]
    observations = torch.stack([trajectory[0] for trajectory in trajectories], dim=0)
    actions = torch.stack([trajectory[1] for trajectory in trajectories], dim=0)
    rewards = torch.stack([trajectory[2] for trajectory in trajectories], dim=0)

    return observations, actions, rewards


def train(policy: Policy, epochs: int = 20):
    for epoch in range(epochs):
        print("Epoch", epoch)
        trajectories = collect_trajectories(policy, H=800)
        policy.train(trajectories)
    policy.save("model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--epochs", type=int, default=20, required=False)
    args = parser.parse_args()

    policy = Policy(env, obs_dim, act_dim)
    policy.model.to(device)

    if args.load:
        print("Loading model...")
        policy.load("model.pt")
    print(f"Training for {args.epochs} epochs...")

    train(policy, args.epochs)
    env.close()
