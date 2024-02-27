"""
Runs pong.
"""

import gym
import numpy as np
import torch
from torchtyping import TensorType

from policy import Policy
from preprocess import preprocess


env = gym.make("ALE/Pong-v5", full_action_space=False)

obs_dim = preprocess(np.zeros((env.observation_space.shape))).numel()
act_dim = env.action_space.n

policy = Policy(env, obs_dim, act_dim)


def collect_trajectory(policy: Policy, H: int = 100):
    observations = []
    actions = []
    rewards = []

    observation, info = env.reset()
    prev_observation = preprocess(observation)
    observation = preprocess(observation)
    for _ in range(H):
        action = policy.act(observation.flatten())
        observation, reward, terminated, truncated, _ = env.step(action)
        observation = preprocess(observation)

        observations.append(observation - prev_observation)
        actions.append(action)
        rewards.append(reward)

        prev_observation = observation

        if terminated or truncated:
            break

    observations_content = torch.stack(observations, dim=0)
    observations = torch.zeros((H, *observations_content.shape[1:]))
    observations[: observations_content.shape[0]] = observations_content

    actions_content = torch.tensor(actions)
    actions = torch.zeros(H, dtype=torch.int64)
    actions[: actions_content.shape[0]] = actions_content

    rewards_content = torch.tensor(rewards)
    rewards = torch.zeros(H, dtype=torch.float32)
    rewards[: rewards_content.shape[0]] = rewards_content

    return observations, actions, rewards


def collect_trajectories(policy: Policy, N: int = 30, H: int = 100) -> tuple[
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
    trajectories = [collect_trajectory(policy, H) for _ in range(N)]
    observations = torch.stack([trajectory[0] for trajectory in trajectories], dim=0)
    actions = torch.stack([trajectory[1] for trajectory in trajectories], dim=0)
    rewards = torch.stack([trajectory[2] for trajectory in trajectories], dim=0)

    return observations, actions, rewards


def train():
    EPOCHS = 20
    for epoch in range(EPOCHS):
        print("Epoch", epoch)
        trajectories = collect_trajectories(policy, H=800)
        policy.train(trajectories)
    policy.save("model.pt")


train()

env.close()
