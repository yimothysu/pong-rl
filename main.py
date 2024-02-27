"""
Runs pong.
"""

import gym
import torch
from torchtyping import TensorType

from policy import Policy

env = gym.make("ALE/Pong-v5", full_action_space=False)

obs_dim = torch.tensor(list(env.observation_space.shape)).prod()
act_dim = env.action_space.n

policy = Policy(env, obs_dim, act_dim)


def collect_trajectory(policy: Policy, H: int = 100):
    observations = []
    actions = []
    rewards = []

    observation, info = env.reset()
    observation = torch.from_numpy(observation).to(torch.float32) / 255.0
    for _ in range(H):
        action = policy.act(observation.flatten())
        observation, reward, terminated, truncated, _ = env.step(action)
        observation = torch.from_numpy(observation).to(torch.float32) / 255.0

        observations.append(observation)
        actions.append(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    observations = torch.stack(observations, dim=0)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)

    return observations, actions, rewards


def collect_trajectories(policy: Policy, N: int = 10, H: int = 100) -> tuple[
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


EPOCHS = 10
for epoch in range(EPOCHS):
    print("Epoch", epoch)
    trajectories = collect_trajectories(policy)
    policy.train(trajectories)

env.close()
env = gym.make("ALE/Pong-v5", full_action_space=False, render_mode="human")
for _ in range(10):
    observation, info = env.reset()
    observation = torch.from_numpy(observation).to(torch.float32) / 255.0
    for _ in range(100):
        env.render()
        action = policy.act(observation.flatten())
        observation, reward, terminated, truncated, _ = env.step(action)
        observation = torch.from_numpy(observation).to(torch.float32) / 255.0
        if terminated or truncated:
            break
