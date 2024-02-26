"""
Runs pong.
"""

import gym
import torch
from torchtyping import TensorType

from policy import Policy

env = gym.make("ALE/Pong-v5", full_action_space=False, render_mode="human")

obs_dim = torch.tensor(list(env.observation_space.shape)).prod()
act_dim = env.action_space.n

policy = Policy(env, obs_dim, act_dim)


def collect_trajectories(policy: Policy, N: int = 10) -> TensorType["N", "3", "H"]:
    """
    Collects `N` trajectories from the environment under the specified policy.

    Returns tensor of shape (N, 3, H), where
    - N is the number of trajectories
    - 3 is the number of elements in the tuple (states, actions, rewards)
    - H is the length of the trajectory
    """
    for _ in range(N):
        actions, states, rewards = [], [], []


EPOCHS = 100
for epoch in range(EPOCHS):
    trajectories = collect_trajectories(policy)
    policy.train(trajectories)
